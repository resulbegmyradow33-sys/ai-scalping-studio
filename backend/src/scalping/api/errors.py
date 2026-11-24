"""Error text that is safe to return over HTTP.

Filesystem and model-loading exceptions embed the absolute path they tried to
open: ``FileNotFoundError("[Errno 2] ... : '/Users/.../runs/abc/model.zip'")``
or a loader's own ``f"no model.zip in {directory}"``. Returning that string to a
client turns a dashboard error into a map of the host's directory layout.

:func:`public_error_text` keeps the reason and replaces the configured root (and
any other absolute path it still sees) with a placeholder. The full exception is
logged where it happens; this helper is only for the response body.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Final

__all__ = ["public_error_text"]

#: Two or more slash-separated segments at a boundary, so a ratio such as
#: ``6/8`` and a relative tail such as ``data/imports/x.csv`` are left alone
#: while ``/etc/passwd`` and ``/var/folders/...`` are not. The POSIX form is
#: what this project runs on; no ``C:\\`` handling is claimed.
_ABSOLUTE_PATH_RE: Final[re.Pattern[str]] = re.compile(r"(?<![\w.\-])(?:/[\w.\-]+){2,}")


def public_error_text(exc: BaseException, *, root: Path | None = None) -> str:
    """``exc``'s message with filesystem paths replaced by stable placeholders.

    ``root`` is the store the call was operating on. Replacing it first is exact
    for the common case (every artefact path starts with it) and survives paths
    containing spaces; the regex is a fallback for anything outside that root.
    """
    text = str(exc)
    if root is not None:
        base = Path(root)
        candidates = {str(base.resolve())}
        # A bare relative name such as ``data`` would rewrite unrelated words in
        # the message; the resolved form above still covers the path the OS
        # reports for the default configuration.
        if base.is_absolute() or len(base.parts) > 1:
            candidates.add(str(base))
        for prefix in sorted(candidates, key=len, reverse=True):
            text = text.replace(prefix, "<root>")
    scrubbed = _ABSOLUTE_PATH_RE.sub("<path>", text)
    # A tail below the configured root is already anonymous; collapsing the two
    # placeholders keeps the message readable ("<root>" not "<root><path>").
    return scrubbed.replace("<root><path>", "<root>")
