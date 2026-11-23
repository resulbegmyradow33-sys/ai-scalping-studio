"""Disabled MetaTrader 5 demo adapter skeleton.

This module is a skeleton, not an integration. It exists so the adapter boundary
is named in code and can be reviewed, and so nothing else in the package grows a
hidden dependency on a terminal. It is disabled by default: constructing
:class:`Mt5DemoAdapter` without ``enabled=True`` raises :class:`AdapterDisabled`,
and even an explicitly enabled instance refuses every operation.

What it does not do:

* no network call, no terminal process, no login and no stored credentials;
* the ``MetaTrader5`` package is loaded lazily inside :meth:`Mt5DemoAdapter.enable`
  and only to prove the dependency is present — importing this module never
  touches it, and a machine without the package gets :class:`AdapterDisabled`;
* every order and account method raises ``NotImplementedError``. Real-money
  execution stays out of scope for the project; the demo path is not built yet.

The paper path in :mod:`scalping.paper.ledger` and
:mod:`scalping.paper.runner` is the only execution model that exists.
"""

from __future__ import annotations

import importlib
from typing import Any, Final, NoReturn

__all__ = ["ORDER_METHODS", "AdapterDisabled", "Mt5DemoAdapter"]

#: The calls that would move money or read an account. Every one of them refuses.
ORDER_METHODS: Final[tuple[str, ...]] = (
    "connect",
    "account_info",
    "open_positions",
    "place_order",
    "modify_order",
    "cancel_order",
    "close_position",
)

#: One message for every refusal, so a caller cannot mistake a stub for success.
_REFUSAL: Final[str] = (
    "the MT5 adapter is a disabled skeleton: no demo session, no order routing and no"
    " real-money execution is implemented; real-money execution stays out of scope"
)


class AdapterDisabled(RuntimeError):
    """The adapter cannot be used; nothing was connected and nothing was sent."""


class Mt5DemoAdapter:
    """A refused-by-construction demo adapter.

    ``enabled`` is the only way past the constructor. It still does not enable
    trading: :meth:`enable` loads the terminal package and stops there, and every
    method in :data:`ORDER_METHODS` raises. There is no state a caller can set to
    make this class place an order.
    """

    def __init__(self, *, enabled: bool = False) -> None:
        if not enabled:
            raise AdapterDisabled(
                "Mt5DemoAdapter is disabled by default; pass enabled=True to load the"
                " demo skeleton, which still refuses every operation"
            )
        self._package: Any = None
        self.enable()

    def enable(self) -> None:
        """Load the terminal package, then stop at the skeleton boundary.

        The import is here rather than at module scope, so this module can be
        imported on any machine. It opens no socket, starts no terminal and reads
        no credentials; it only proves the package is installed. A missing
        package is :class:`AdapterDisabled`, never a raw ``ImportError``.
        """
        try:
            self._package = importlib.import_module("MetaTrader5")
        except ImportError as exc:
            raise AdapterDisabled(
                "MetaTrader5 is not installed; the demo adapter is a skeleton and cannot"
                " connect, and real-money execution stays out of scope"
            ) from exc

    @property
    def enabled(self) -> bool:
        """Whether the package was found; it never means trading is possible."""
        return self._package is not None

    # ------------------------------------------------------------------ refused

    def connect(self, *args: Any, **kwargs: Any) -> NoReturn:
        """Refused: no terminal connection exists in this slice."""
        raise NotImplementedError(_REFUSAL)

    def account_info(self, *args: Any, **kwargs: Any) -> NoReturn:
        """Refused: there is no account to read without a connection."""
        raise NotImplementedError(_REFUSAL)

    def open_positions(self, *args: Any, **kwargs: Any) -> NoReturn:
        """Refused: the adapter holds no positions."""
        raise NotImplementedError(_REFUSAL)

    def place_order(self, *args: Any, **kwargs: Any) -> NoReturn:
        """Refused: real-money execution is out of scope, and no demo order path exists."""
        raise NotImplementedError(_REFUSAL)

    def modify_order(self, *args: Any, **kwargs: Any) -> NoReturn:
        """Refused: no order can exist to modify."""
        raise NotImplementedError(_REFUSAL)

    def cancel_order(self, *args: Any, **kwargs: Any) -> NoReturn:
        """Refused: no order can exist to cancel."""
        raise NotImplementedError(_REFUSAL)

    def close_position(self, *args: Any, **kwargs: Any) -> NoReturn:
        """Refused: no position can exist to close."""
        raise NotImplementedError(_REFUSAL)
