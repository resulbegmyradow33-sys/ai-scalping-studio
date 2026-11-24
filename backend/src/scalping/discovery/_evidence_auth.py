"""Private capability between the promotion policy and the discovery archive.

The archive must not promote evidence on its own. Only an evaluation that
already produced a sealed :class:`~scalping.discovery.promotion.PromotionDecision`
may call the private archive primitive, and the archive needs a way to check
that. This module holds a per-process HMAC key; the promotion module signs the
promotion identity, the archive verifies it and compares the signed fields with
its own arguments.

This is not a cryptographic boundary against code running in the same process —
nothing in Python can be — but it turns "call the private method with plausible
strings" into a deliberate private-API abuse instead of an ordinary call.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
from collections.abc import Mapping
from typing import Any

from scalping.discovery.protocols import canonical_json

__all__: list[str] = []

_KEY = secrets.token_bytes(32)


def issue(payload: Mapping[str, Any]) -> str:
    """Sign a promotion identity for the archive."""
    return hmac.new(_KEY, canonical_json(payload).encode("utf-8"), hashlib.sha256).hexdigest()


def verify(payload: Mapping[str, Any], tag: object) -> bool:
    """Constant-time verification of a promotion identity tag."""
    if not isinstance(tag, str):
        return False
    return hmac.compare_digest(issue(payload), tag)
