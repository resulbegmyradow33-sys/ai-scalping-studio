"""Risk engine: dual-limit sizing, session gates, kill switches and ATR stops.

The engine owns the FX-day session anchor and the hard risk limits. Every limit
is frozen at construction: ``size``/``allow_new`` take prices and an account
snapshot, never a risk override, so a caller cannot loosen a limit per trade.
There is no per-trade or per-policy override argument anywhere in this module:
a model can choose an action, not the limits applied to it. This is the
Phase 11 "risk blocks cannot be bypassed by a policy" rule.

Session anchor
--------------
``session_anchor`` returns the ``date`` the 17:00 roll belongs to: a bar at
or after 17:00 UTC belongs to that calendar day, a bar before 17:00 UTC belongs
to the previous one. It is the single boundary the Simulator uses to roll
``day_start_equity``. The design calls for the New York 17:00 roll; v1 uses
17:00 UTC as a fixed approximation and ignores US daylight saving, which shifts
the true roll by one hour for part of the year.

``week_anchor`` reuses that session boundary and labels each session with its
ISO year/week. The FX week therefore rolls at the first 17:00 UTC session
anchor whose ISO week differs from the previous one; with the fixed 17:00 UTC
approximation that is Monday 17:00 UTC.

Kill switches
-------------
All Phase 11 limits are disabled at their default ``0`` and the engine never
invents a missing anchor. Semantics, all condition-driven unless ``pause_bars``
latches them:

* ``weekly_loss_pct`` — loss from ``week_start_equity`` at or beyond the
  fraction. Blocks entries and force-closes; clears when the week anchor rolls.
* ``max_drawdown_pct`` — ``(peak_equity - equity) / peak_equity`` at or beyond
  the fraction. Blocks entries and force-closes; clears when equity recovers.
* ``min_equity_frac`` — emergency stop when equity is strictly below
  ``min_equity_frac * ContractSpec.initial_balance``. Blocks entries and
  force-closes; clears when equity recovers.
* ``consecutive_loss_pause`` — blocks entries once
  ``consecutive_losses >= consecutive_loss_pause``. The Simulator resets the
  count when the pause expires, or at the next session roll when
  ``pause_bars == 0``.
* ``max_spread_points`` — blocks entries when ``spread_points`` strictly
  exceeds the cap. Per-bar only; never latched.
* ``pause_bars`` — when positive, a tripped weekly/drawdown/emergency or
  consecutive-loss switch stays latched for that many decision bars after the
  trip bar (entries resume on the bar whose timestamp equals ``paused_until``).
  ``0`` leaves the switch condition-driven only.

The engine decides from the passed :class:`AccountState`, not from broker
internals. That snapshot is expected to be anchored to the session containing
``ts`` (``day_start_equity``/``week_start_equity`` are the equity at their
anchors, ``peak_equity`` the episode peak), so every gate is a pure function of
the snapshot.
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


def _check_optional_fraction(name: str, value: float) -> None:
    """Validate a switch fraction: ``0`` disables it, otherwise ``(0, 1]``."""
    _check_finite(name, value)
    if not 0 <= value <= 1:
        raise ValueError(f"{name} must be 0 or in (0, 1], got {value!r}")


def _check_optional_non_negative(name: str, value: float) -> None:
    """Validate a switch threshold where ``0`` disables it."""
    _check_finite(name, value)
    if value < 0:
        raise ValueError(f"{name} must be >= 0, got {value!r}")


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
            raise ValueError(f"risk.max_bars_in_trade must be >= 1, got {risk.max_bars_in_trade!r}")
        _check_optional_fraction("risk.weekly_loss_pct", risk.weekly_loss_pct)
        _check_optional_fraction("risk.max_drawdown_pct", risk.max_drawdown_pct)
        _check_optional_fraction("risk.min_equity_frac", risk.min_equity_frac)
        _check_optional_non_negative("risk.max_spread_points", risk.max_spread_points)
        if risk.consecutive_loss_pause < 0:
            raise ValueError(
                f"risk.consecutive_loss_pause must be >= 0, got {risk.consecutive_loss_pause!r}"
            )
        if risk.pause_bars < 0:
            raise ValueError(f"risk.pause_bars must be >= 0, got {risk.pause_bars!r}")
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

        Backward-compatible entry point: it delegates to
        :meth:`allow_new_extended`. With every Phase 11 limit at its default
        ``0`` the decisions are identical to the pre-Phase-11 engine, because
        each new check is gated on its own configured limit.

        Checks run in this order: a non-positive ``day_start_equity``
        (``"lot_too_small"``: nothing can be measured or sized), the daily loss
        limit at or beyond ``daily_loss_pct`` (``"daily_loss_limit"``; zero or
        negative equity on a real base is a total loss, not a sizing failure),
        then the position count (``"max_positions"``), then the Phase 11
        switches. The account snapshot is never mutated.
        """
        return self.allow_new_extended(account, ts)

    def allow_new_extended(self, account: AccountState, ts: datetime) -> tuple[bool, str | None]:
        """Full entry gate, including every Phase 11 kill switch.

        Order after the legacy checks: weekly loss, max drawdown, emergency
        equity floor, consecutive losses, max spread, then a latched
        ``paused_until``. A missing anchor (``None``) or a disabled limit
        (``0``) skips that check; the engine never invents an anchor.
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

        kill = self.kill_switch_reason(account)
        if kill is not None:
            return False, kill
        if self._consecutive_losses_tripped(account):
            return False, "consecutive_loss_pause"
        if self._spread_tripped(account):
            return False, "max_spread"
        if self._paused(account, ts):
            return False, "kill_switch_pause"
        return True, None

    def kill_switch_reason(self, account: AccountState) -> str | None:
        """The equity kill switch that is currently engaged, or ``None``.

        Covers the three switches that force-close an open position:
        ``"weekly_loss_limit"``, ``"max_drawdown"`` and ``"emergency_stop"``.
        Each is skipped when its limit is ``0`` or when the anchor it needs is
        missing or non-positive.
        """
        _check_finite("account.equity", account.equity)
        risk = self._risk
        if risk.weekly_loss_pct > 0 and account.week_start_equity is not None:
            anchor = account.week_start_equity
            _check_finite("account.week_start_equity", anchor)
            if anchor > 0 and (anchor - account.equity) / anchor >= risk.weekly_loss_pct:
                return "weekly_loss_limit"
        if risk.max_drawdown_pct > 0 and account.peak_equity is not None:
            peak = account.peak_equity
            _check_finite("account.peak_equity", peak)
            if peak > 0 and (peak - account.equity) / peak >= risk.max_drawdown_pct:
                return "max_drawdown"
        if risk.min_equity_frac > 0:
            floor = risk.min_equity_frac * self._contract.initial_balance
            if account.equity < floor:
                return "emergency_stop"
        return None

    def _consecutive_losses_tripped(self, account: AccountState) -> bool:
        limit = self._risk.consecutive_loss_pause
        return limit > 0 and account.consecutive_losses >= limit

    def _spread_tripped(self, account: AccountState) -> bool:
        limit = self._risk.max_spread_points
        if limit <= 0 or account.spread_points is None:
            return False
        _check_finite("account.spread_points", account.spread_points)
        return account.spread_points > limit

    def _paused(self, account: AccountState, ts: datetime) -> bool:
        """True while a ``pause_bars`` latch is still engaged.

        ``paused_until`` is honoured only when ``pause_bars > 0``; with the
        default the field is inert, matching the pre-Phase-11 engine.
        """
        if self._risk.pause_bars <= 0 or account.paused_until is None:
            return False
        paused_until = account.paused_until
        if paused_until.tzinfo is None or paused_until.utcoffset() is None:
            raise ValueError(f"account.paused_until must be timezone-aware, got {paused_until!r}")
        return ts < paused_until

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
        return (
            entry - direction * atr * self.risk.sl_atr_mult,
            entry + self.risk.tp_atr_mult * direction * atr,
        )

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

    def week_anchor(self, ts: datetime) -> tuple[int, int]:
        """Return the ``(iso_year, iso_week)`` of the session containing ``ts``.

        Built on :meth:`session_anchor`, so the week rolls on the same 17:00 UTC
        boundary as the day: the first session anchor whose ISO week differs
        starts a new week. With the fixed approximation that is Monday 17:00
        UTC. A naive datetime is rejected by ``session_anchor``.
        """
        iso = self.session_anchor(ts).isocalendar()
        return iso.year, iso.week
