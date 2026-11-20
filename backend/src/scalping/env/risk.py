"""Risk engine: dual-limit sizing, session gates and ATR stop placement.

The engine owns the FX-day session anchor and the hard risk limits. Every limit
is frozen at construction: ``size``/``allow_new`` take prices and an account
snapshot, never a risk override, so a caller cannot loosen a limit per trade.

Session anchor
--------------
``session_anchor`` returns the ``date`` the 17:00 roll belongs to: a bar at
or after 17:00 UTC belongs to that calendar day, a bar before 17:00 UTC belongs
to the previous one. It is the single boundary the Simulator uses to roll
``day_start_equity``. The design calls for the New York 17:00 roll; v1 uses
17:00 UTC as a fixed approximation and ignores US daylight saving, which shifts
the true roll by one hour for part of the year.

The engine decides from the passed :class:`AccountState`, not from broker
internals. That snapshot is expected to be anchored to the session containing
``ts`` (``day_start_equity`` is the equity at the anchor), so the daily loss
gate is a pure function of the snapshot.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from math import floor, isfinite

from scalping.types import AccountState, ContractSpec, RiskConfig, Side

__all__ = ["RiskEngine", "SizingResult"]

#: Absorbs binary representation error when flooring lots onto ``lot_step``
#: (e.g. ``0.29 / 0.01`` is ``28.999999999999996`` before the floor).
_STEP_EPS = 1e-9

#: Decimal places used to clean floored lots (``85 * 0.01`` is not 0.85).
_LOT_DP = 10


def _check_finite(name: str, value: float) -> None:
    if not isfinite(value):
        raise ValueError(f"{name} must be finite, got {value!r}")


def _check_positive(name: str, value: float) -> None:
    _check_finite(name, value)
    if value <= 0:
        raise ValueError(f"{name} must be > 0, got {value!r}")


def _check_fraction(name: str, value: float) -> None:
    """Validate a fraction in ``(0, 1]`` (risk per trade, daily loss limit)."""
    _check_finite(name, value)
    if not 0 < value <= 1:
        raise ValueError(f"{name} must be in (0, 1], got {value!r}")


@dataclass(frozen=True, slots=True)
class SizingResult:
    """Outcome of one sizing decision.

    ``reason`` is ``None`` when the risk budget binds, ``"notional_cap"`` when
    the leverage cap is the tighter limit, and ``"lot_too_small"`` when the
    floored lot falls below ``ContractSpec.min_lot`` (``lot`` is then ``0.0``).
    """

    lot: float
    reason: str | None
    effective_risk_pct: float


class RiskEngine:
    """Position sizing and entry gates for a single-account scalping run."""

    def __init__(self, risk: RiskConfig, contract: ContractSpec) -> None:
        _check_fraction("risk.risk_pct", risk.risk_pct)
        _check_fraction("risk.daily_loss_pct", risk.daily_loss_pct)
        _check_positive("risk.max_notional_x", risk.max_notional_x)
        if risk.max_positions < 1:
            raise ValueError(f"risk.max_positions must be >= 1, got {risk.max_positions!r}")
        if risk.max_bars_in_trade < 1:
            raise ValueError(
                f"risk.max_bars_in_trade must be >= 1, got {risk.max_bars_in_trade!r}"
            )
        _check_positive("contract.size", contract.size)
        _check_positive("contract.min_lot", contract.min_lot)
        _check_positive("contract.lot_step", contract.lot_step)
        self._risk = risk
        self._contract = contract

    # --- configuration -----------------------------------------------------

    @property
    def risk(self) -> RiskConfig:
        """The frozen risk limits; rebinding the attribute is rejected."""
        return self._risk

    @property
    def contract(self) -> ContractSpec:
        """The frozen contract metadata; rebinding the attribute is rejected."""
        return self._contract

    # --- sizing ------------------------------------------------------------

    def size(self, equity: float, entry: float, sl: float) -> SizingResult:
        """Return the lot for one trade at the tighter of the two limits.

        ``lot_risk = equity * risk_pct / (|entry - sl| * size)`` spends the risk
        budget exactly, while ``lot_cap = equity * max_notional_x / (entry *
        size)`` respects the leverage cap. The result is floored to
        ``lot_step``; when both lots are equal the cap does not bind.
        """
        _check_finite("equity", equity)
        _check_finite("entry", entry)
        _check_finite("sl", sl)
        if entry <= 0:
            raise ValueError(f"entry must be > 0, got {entry!r}")
        distance = abs(entry - sl)
        if distance == 0:
            raise ValueError("entry and sl must differ")
        if equity <= 0:
            # Nothing to risk: report the sizing reason instead of dividing by zero.
            return SizingResult(lot=0.0, reason="lot_too_small", effective_risk_pct=0.0)

        size = self._contract.size
        lot_risk = equity * self._risk.risk_pct / (distance * size)
        lot_cap = equity * self._risk.max_notional_x / (entry * size)
        cap_binds = lot_cap < lot_risk
        lot = self._floor_lot(lot_cap if cap_binds else lot_risk)
        if lot < self._contract.min_lot:
            return SizingResult(lot=0.0, reason="lot_too_small", effective_risk_pct=0.0)

        effective = lot * distance * size / equity
        return SizingResult(
            lot=lot,
            reason="notional_cap" if cap_binds else None,
            effective_risk_pct=effective,
        )

    def _floor_lot(self, raw: float) -> float:
        """Floor a raw lot onto the contract's ``lot_step`` without rounding up."""
        step = self._contract.lot_step
        steps = floor(raw / step + _STEP_EPS)
        return round(steps * step, _LOT_DP)

    # --- entry gate --------------------------------------------------------

    def allow_new(self, account: AccountState, ts: datetime) -> tuple[bool, str | None]:
        """Decide whether a new position may open at ``ts``.

        Checks run in this order: a non-positive ``day_start_equity``
        (``"lot_too_small"``: nothing can be measured or sized), the daily loss
        limit at or beyond ``daily_loss_pct`` (``"daily_loss_limit"``; zero or
        negative equity on a real base is a total loss, not a sizing failure),
        then the position count (``"max_positions"``). The account snapshot is
        never mutated.
        """
        # The snapshot is expected to be anchored to the session ``ts`` names.
        # Computing the anchor validates ts (aware, normalised to UTC) and makes
        # the session boundary explicit; the gate itself needs only the equity
        # pair, because the caller rolls ``day_start_equity`` on this boundary.
        self.session_anchor(ts)
        _check_finite("account.equity", account.equity)
        _check_finite("account.day_start_equity", account.day_start_equity)
        if account.day_start_equity <= 0:
            return False, "lot_too_small"

        loss_pct = (account.day_start_equity - account.equity) / account.day_start_equity
        if loss_pct >= self._risk.daily_loss_pct:
            return False, "daily_loss_limit"

        open_positions = 1 if account.position is not None else 0
        if open_positions >= self._risk.max_positions:
            return False, "max_positions"
        return True, None

    # --- exits -------------------------------------------------------------

    def must_force_close(self, bars_in_trade: int) -> bool:
        """True once the position has been held for ``max_bars_in_trade`` bars."""
        return bars_in_trade >= self._risk.max_bars_in_trade

    def sl_tp(self, entry: float, side: Side, atr: float) -> tuple[float, float]:
        """Place a 1.0x ATR stop and a 2.0x ATR target around ``entry``."""
        _check_finite("entry", entry)
        _check_finite("atr", atr)
        if atr <= 0:
            raise ValueError(f"atr must be > 0, got {atr!r}")
        side = Side(side)
        if side is Side.FLAT:
            raise ValueError("side must be LONG or SHORT")
        direction = int(side)
        return entry - direction * atr, entry + 2 * direction * atr

    # --- session -----------------------------------------------------------

    def session_anchor(self, ts: datetime) -> date:
        """Return the 17:00 UTC FX-day ``ts`` belongs to.

        This is the one session boundary in the backend: Task 10 rolls
        ``day_start_equity`` when this anchor changes. v1 approximates the New
        York 17:00 roll with a fixed 17:00 UTC boundary and ignores daylight
        saving, so the anchor is the date of the most recent 17:00 UTC boundary:
        ``ts`` at or after 17:00 anchors to its own date, earlier than that to
        the previous date. A naive datetime is rejected rather than silently
        read as local time.
        """
        if ts.tzinfo is None or ts.utcoffset() is None:
            raise ValueError(f"ts must be timezone-aware, got {ts!r}")
        utc = ts.astimezone(UTC)
        if utc.hour >= 17:
            return utc.date()
        return utc.date() - timedelta(days=1)
