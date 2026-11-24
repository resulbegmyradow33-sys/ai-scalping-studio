"""Market-data files as ``dataset.build`` triggers (Phase 18).

The orchestrator can already build a dataset for one ``symbol:timeframe`` and
the queue already dedupes a job by key. This module is the missing edge: when a
new or changed Parquet/CSV file lands under the data root, enqueue exactly one
``dataset.build`` for the symbol and timeframe the *layout* names. Nothing here
opens a market feed, schedules a loop or reads bar rows; it watches a directory
the owner fills.

Contract
--------
:class:`DataFileTriggerSource` implements the same two-phase
:class:`~scalping.orchestrator.triggers.TriggerSource` protocol as
``FileTriggerSource``:

* ``scan()`` returns one :class:`~scalping.orchestrator.triggers.Trigger` per
  file whose *content* this root has not emitted before. The payload is the
  ``dataset.build`` contract — ``symbol``, ``timeframe`` and an audit ``notes``
  string — so the registered handler accepts it unchanged. No raw file fields
  (``path``, ``size``, ``hash``) are mixed in: that payload shape belongs to a
  different handler and is rejected here.
* ``seen(triggers)`` persists the cursor after the orchestrator enqueued the
  jobs. The cursor lives in the job store's ``state`` table, so the queue and
  its cursors survive or are lost together.

Dedupe has two layers
---------------------
The cursor records, per relative path, the last ``(size, mtime_ns)`` seen plus
the file's full SHA-256, and keeps a set of every checksum already emitted. A
stat match skips the file without hashing; a digest already in the emitted set
updates the cursor but produces no trigger. The job's ``dedupe_key`` is
``dataset.build:data:{sha256 prefix}`` as the second layer: a crash between
``scan`` and ``seen`` re-emits the same trigger on the next pass and the store
returns the existing job instead of a duplicate. A changed file has a new digest
and therefore emits once; a rewrite with identical bytes is a duplicate.

Layout inference
----------------
The only accepted shape is ``<symbol>/<timeframe>/.../<file>.parquet|.csv``,
where ``symbol`` matches the same token rule the ``dataset.build`` handler
enforces and ``timeframe`` is one of :data:`scalping.data.registry.TIMEFRAMES`
(case-insensitively, normalised to its canonical upper-case key). ``symbol`` is
upper-cased the same way the data registry already looks it up. Deeper
directories are allowed; the store's own ``<year>/<month>.parquet`` partitions
match without a special case.

A file that does not match is *refused*, not guessed: ``data.trigger_refused``
is written with the reason (``unknown_layout``, ``invalid_symbol`` or
``unknown_timeframe``) and the cursor records the refusal so a periodic scan
does not repeat the same line. Renaming the file into a valid layout makes it a
new relative path and it is evaluated normally.

Limits
------
* No scheduler thread, no filesystem watcher, no market clock. ``scan`` is a
  call. Whether it happens once, every five seconds or after a cron tick is the
  caller's decision. A file written while the scan is running may be picked up
  on the next pass.
* ``mtime`` is filesystem metadata, not event time. A copy, restore or network
  filesystem can preserve a stale mtime, and a clock change can move it
  backwards; ``since`` and the stat fast path both trust it. A file whose mtime
  is older than ``since`` is skipped until it changes.
* Extension and layout are the only signal. A stray CSV under a valid
  symbol/timeframe triggers a rebuild; the handler, not this source, decides
  whether the stored bars are readable. Content is never parsed here.
* ``max_emits`` caps one scan so a first run over a large historical root
  cannot enqueue an unbounded backlog at once. Files over the cap are left out
  of the persisted cursor and are re-evaluated on the next scan, not lost.
* ``discovery.run`` is deliberately not emitted. Its payload requires a gate,
  a trial budget and a seed, none of which a filename can supply; inventing
  them would turn a data arrival into an unapproved search policy.
* Dot-prefixed path segments are ignored. That skips the ``.feature-cache``
  directory and atomic-write temp files without a special-case list.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Final

from scalping.data.registry import TIMEFRAMES
from scalping.orchestrator.store import JobStore
from scalping.orchestrator.triggers import Trigger

__all__ = [
    "CHECKSUM_PREFIX_LENGTH",
    "EVENT_DEFERRED",
    "EVENT_REFUSED",
    "KIND_DATASET_BUILD",
    "MAX_EMITS",
    "MAX_EMITS_LIMIT",
    "REASON_INVALID_SYMBOL",
    "REASON_UNKNOWN_LAYOUT",
    "REASON_UNKNOWN_TIMEFRAME",
    "DataFileTriggerSource",
]

#: The kind this source produces. Kept as a literal so importing this module
#: does not pull the handler stack (discovery, paper, polars) into a trigger-only
#: process; ``test_data_triggers`` asserts it equals the handler's constant.
KIND_DATASET_BUILD: Final[str] = "dataset.build"

#: Audit event for a file the layout cannot place. It carries no work.
EVENT_REFUSED: Final[str] = "data.trigger_refused"
#: Audit event when ``max_emits`` cut a scan short. The files are retried.
EVENT_DEFERRED: Final[str] = "data.trigger_deferred"

#: Refusal reasons. They are stable strings, not prose, so the audit page can
#: filter them; the human detail goes in the same event.
REASON_UNKNOWN_LAYOUT: Final[str] = "unknown_layout"
REASON_INVALID_SYMBOL: Final[str] = "invalid_symbol"
REASON_UNKNOWN_TIMEFRAME: Final[str] = "unknown_timeframe"

#: Upper bound on triggers from one scan. A first scan of a decade of month
#: partitions is a backlog, not a useful queue.
MAX_EMITS: Final[int] = 100
MAX_EMITS_LIMIT: Final[int] = 1_000

#: Hex characters of SHA-256 used in the job's dedupe key. The cursor stores the
#: full digest for audit; 64 bits is enough to keep two different files apart
#: while keeping the key short and readable.
CHECKSUM_PREFIX_LENGTH: Final[int] = 16

#: One retry by default: a transient store failure must not lose the only
#: trigger for a file that is already on disk.
DEFAULT_MAX_RETRIES: Final[int] = 1

#: Files the scan considers. Everything else is ignored rather than refused:
#: the data root also holds caches and metadata written by other stages.
_SUFFIXES: Final[frozenset[str]] = frozenset({".parquet", ".csv"})

#: Same token rule as the ``dataset.build`` payload validator. A symbol is a
#: path segment, so it must not be able to escape the root.
_SYMBOL_RE: Final[re.Pattern[str]] = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,31}$")

_CURSOR_VERSION: Final[int] = 1
_CHUNK_BYTES: Final[int] = 1 << 20


@dataclass(frozen=True, slots=True)
class _Layout:
    """A path the source is willing to turn into a ``dataset.build``."""

    symbol: str
    timeframe: str


@dataclass(frozen=True, slots=True)
class _Refusal:
    """Why a path is not a dataset trigger; carries no work."""

    reason: str
    detail: str


class DataFileTriggerSource:
    """Watch a market-data root and enqueue one ``dataset.build`` per new file.

    ``root`` is the same directory a ``ParquetStore`` reads: it may not exist
    yet, in which case ``scan`` is a no-op rather than an error, because a fresh
    install routinely configures the watch before the first download. A path
    that exists but is not a directory is a configuration mistake and raises
    ``NotADirectoryError`` in the constructor.

    ``since`` bounds the scan by file mtime; ``max_emits`` bounds one scan. The
    cursor is stored under ``trigger.data:<kind>:<absolute root>`` in the job
    store, so constructing a second source for the same root reuses it.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        store: JobStore,
        kind: str = KIND_DATASET_BUILD,
        priority: int = 0,
        max_retries: int = DEFAULT_MAX_RETRIES,
        max_emits: int = MAX_EMITS,
        since: str | datetime | None = None,
        name: str | None = None,
    ) -> None:
        self.root = Path(root)
        if self.root.exists() and not self.root.is_dir():
            raise NotADirectoryError(f"{self.root} is not a directory")
        if str(kind) != KIND_DATASET_BUILD:
            raise ValueError(
                f"kind must be {KIND_DATASET_BUILD!r}; the payload this source emits"
                " has no other registered handler"
            )
        if isinstance(max_emits, bool) or not isinstance(max_emits, int):
            raise TypeError("max_emits must be an int")
        if max_emits < 1:
            raise ValueError("max_emits must be >= 1")
        if max_emits > MAX_EMITS_LIMIT:
            raise ValueError(f"max_emits must be <= {MAX_EMITS_LIMIT}")
        if isinstance(max_retries, bool) or not isinstance(max_retries, int):
            raise TypeError("max_retries must be an int")
        if max_retries < 0:
            raise ValueError("max_retries must be >= 0")
        self.kind = KIND_DATASET_BUILD
        self.store = store
        self.priority = int(priority)
        self.max_retries = max_retries
        self.max_emits = max_emits
        self.since = _coerce_since(since)
        self.name = name or f"data:{self.root}"
        self.cursor_key = f"trigger.data:{self.kind}:{self.root.resolve()}"
        self._pending: dict[str, object] | None = None

    # ------------------------------------------------------------------ protocol

    def scan(self) -> list[Trigger]:
        """Files the cursor has not emitted, in relative-path order."""
        self._pending = None
        if not self.root.is_dir():
            return []
        cursor = self._load_cursor()
        files = _file_map(cursor.get("files"))
        checksums = _text_map(cursor.get("checksums"))
        pending: dict[str, object] = {}
        triggers: list[Trigger] = []
        deferred = 0

        for path in self._list_files():
            try:
                stat = path.stat()
            except FileNotFoundError:
                continue  # removed between listing and stat
            relative = path.relative_to(self.root).as_posix()
            size = int(stat.st_size)
            mtime_ns = int(stat.st_mtime_ns)
            entry = files.get(relative)
            if entry is not None and _stat_matches(entry, size=size, mtime_ns=mtime_ns):
                pending[relative] = dict(entry)
                continue

            layout = _infer_layout(relative)
            if isinstance(layout, _Refusal):
                pending[relative] = {
                    "size": size,
                    "mtime_ns": mtime_ns,
                    "status": "refused",
                    "reason": layout.reason,
                }
                self._refuse(relative, layout)
                continue

            if self.since is not None and _mtime(stat.st_mtime) < self.since:
                # Bounded history: record the stat so the next scan does not
                # re-stat a file this source is configured to ignore. A later
                # write moves the mtime and it is evaluated again.
                pending[relative] = {"size": size, "mtime_ns": mtime_ns, "status": "before_since"}
                continue

            digest = _sha256(path)
            prefix = digest[:CHECKSUM_PREFIX_LENGTH]
            entry_out: dict[str, object] = {
                "size": size,
                "mtime_ns": mtime_ns,
                "sha256": digest,
                "symbol": layout.symbol,
                "timeframe": layout.timeframe,
                "status": "duplicate" if digest in checksums else "emitted",
            }
            pending[relative] = entry_out
            if digest in checksums:
                # Same bytes under another name, or a reverted file: the work
                # already exists. The cursor still records the new location.
                continue
            if len(triggers) >= self.max_emits:
                # Not persisted, so the next scan sees it as new again. It is
                # delayed by the cap, never dropped.
                pending.pop(relative, None)
                deferred += 1
                continue
            checksums[digest] = relative
            triggers.append(
                Trigger(
                    kind=self.kind,
                    dedupe_key=f"{self.kind}:data:{prefix}",
                    payload={
                        "symbol": layout.symbol,
                        "timeframe": layout.timeframe,
                        "notes": (
                            f"auto-triggered by data file {relative} "
                            f"(sha256 {prefix}, {size} bytes)"
                        ),
                    },
                    priority=self.priority,
                    max_retries=self.max_retries,
                )
            )

        if deferred:
            self.store.append_event(
                EVENT_DEFERRED,
                detail=_json(
                    {
                        "root": str(self.root),
                        "deferred": deferred,
                        "max_emits": self.max_emits,
                    }
                ),
            )
        self._pending = {
            "version": _CURSOR_VERSION,
            "files": pending,
            "checksums": checksums,
        }
        return triggers

    def seen(self, triggers: Sequence[Trigger]) -> None:
        """Persist the cursor built by the last :meth:`scan`.

        The argument is uniform with the protocol. Acknowledgement is per
        source, not per trigger, because ``scan_once`` calls this after the
        whole batch was enqueued; a crash before this call re-emits the batch
        and the job ``dedupe_key`` absorbs the repetition.
        """
        if self._pending is None:
            return
        self.store.set_state(
            self.cursor_key,
            json.dumps(self._pending, sort_keys=True, separators=(",", ":")),
        )
        self._pending = None

    # ------------------------------------------------------------------ internals

    def _list_files(self) -> list[Path]:
        if not self.root.is_dir():
            return []
        found: list[Path] = []
        for path in self.root.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix.lower() not in _SUFFIXES:
                continue
            relative = path.relative_to(self.root)
            if any(part.startswith(".") for part in relative.parts):
                continue  # .feature-cache, atomic-write temp files and dotfiles
            found.append(path)
        return sorted(found, key=lambda item: item.relative_to(self.root).as_posix())

    def _refuse(self, relative: str, refusal: _Refusal) -> None:
        self.store.append_event(
            EVENT_REFUSED,
            detail=_json(
                {
                    "root": str(self.root),
                    "relative_path": relative,
                    "reason": refusal.reason,
                    "detail": refusal.detail,
                }
            ),
        )

    def _load_cursor(self) -> dict[str, object]:
        raw = self.store.get_state(self.cursor_key)
        if raw is None:
            return {}
        try:
            parsed = json.loads(raw)
        except ValueError:
            return {}  # a corrupt cursor costs a re-hash; dedupe still holds
        return parsed if isinstance(parsed, dict) else {}


# --------------------------------------------------------------------- helpers


def _infer_layout(relative: str) -> _Layout | _Refusal:
    """Read ``<symbol>/<timeframe>`` from a relative path, or refuse it."""
    parts = PurePosixPath(relative).parts
    if len(parts) < 3:
        return _Refusal(
            REASON_UNKNOWN_LAYOUT,
            "expected <symbol>/<timeframe>/.../<file>.parquet|.csv",
        )
    symbol_raw, timeframe_raw = parts[0], parts[1]
    if not _SYMBOL_RE.fullmatch(symbol_raw):
        return _Refusal(
            REASON_INVALID_SYMBOL,
            f"symbol segment {symbol_raw!r} is not a valid dataset token",
        )
    timeframe = timeframe_raw.strip().upper()
    if timeframe not in TIMEFRAMES:
        supported = ", ".join(sorted(TIMEFRAMES))
        return _Refusal(
            REASON_UNKNOWN_TIMEFRAME,
            f"timeframe segment {timeframe_raw!r} is not one of {supported}",
        )
    return _Layout(symbol=symbol_raw.upper(), timeframe=timeframe)


def _file_map(value: object) -> dict[str, Mapping[str, object]]:
    if not isinstance(value, Mapping):
        return {}
    return {
        str(name): entry
        for name, entry in value.items()
        if isinstance(name, str) and isinstance(entry, Mapping)
    }


def _text_map(value: object) -> dict[str, str]:
    if not isinstance(value, Mapping):
        return {}
    return {
        str(key): str(item)
        for key, item in value.items()
        if isinstance(key, str) and isinstance(item, str)
    }


def _stat_matches(entry: Mapping[str, object], *, size: int, mtime_ns: int) -> bool:
    return entry.get("size") == size and entry.get("mtime_ns") == mtime_ns


def _mtime(seconds: float) -> datetime:
    return datetime.fromtimestamp(seconds, tz=UTC)


def _coerce_since(value: str | datetime | None) -> datetime | None:
    """Normalise the optional lower bound; ``None`` means the whole root."""
    if value is None:
        return None
    if isinstance(value, datetime):
        moment = value
    elif isinstance(value, str):
        try:
            moment = datetime.fromisoformat(value)
        except ValueError as exc:
            raise ValueError(f"since is not an ISO-8601 timestamp: {value!r}") from exc
    else:
        raise TypeError(
            f"since must be a datetime, an ISO-8601 string or None, got {type(value).__name__}"
        )
    if moment.tzinfo is None or moment.utcoffset() is None:
        raise ValueError("since must be timezone-aware")
    return moment.astimezone(UTC)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(_CHUNK_BYTES)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _json(payload: Mapping[str, object]) -> str:
    return json.dumps(dict(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
