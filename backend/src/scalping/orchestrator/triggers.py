"""What tells the orchestrator there is new work.

A trigger source is a small object with two methods: ``scan`` says what it found,
``seen`` is called after the orchestrator has enqueued those triggers. The two
phases exist because neither order is safe on its own: marking a file seen
before the job exists loses the job on a crash, and enqueuing without a cursor
re-hashes the whole directory on every tick. Crash between the phases is
covered by ``dedupe_key`` — the content hash — so the second attempt returns the
existing job instead of a duplicate.

There is no scheduler thread and no filesystem watcher in this slice. ``scan_once``
is a call, not a loop: a CLI or a future daemon decides how often to make it.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Protocol, runtime_checkable

from scalping.orchestrator.core import Orchestrator
from scalping.orchestrator.store import Job, JobStore

__all__ = [
    "FileTriggerSource",
    "ScanOutcome",
    "Trigger",
    "TriggerSource",
    "scan_once",
]

#: Read size when hashing a file; large enough to keep syscalls cheap.
_CHUNK_BYTES: int = 1 << 20


@dataclass(frozen=True, slots=True)
class Trigger:
    """One reason to enqueue a job, in the shape ``JobStore.enqueue`` wants.

    ``dedupe_key`` is the trigger's identity, not its description: two scans that
    see the same underlying event must produce the same key.
    """

    kind: str
    dedupe_key: str
    payload: Mapping[str, object] = field(default_factory=dict)
    priority: int = 0
    max_retries: int = 0
    depends_on: tuple[int, ...] = ()


@runtime_checkable
class TriggerSource(Protocol):
    """A producer of triggers with an explicit acknowledgement step."""

    name: str

    def scan(self) -> Sequence[Trigger]: ...

    def seen(self, triggers: Sequence[Trigger]) -> None: ...


@dataclass(frozen=True, slots=True)
class ScanOutcome:
    """What one ``scan_once`` call enqueued and what already existed."""

    sources: tuple[str, ...]
    queued: tuple[Job, ...]
    duplicates: tuple[Job, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "sources": list(self.sources),
            "queued": [job.to_dict() for job in self.queued],
            "duplicates": [job.to_dict() for job in self.duplicates],
        }


def scan_once(
    orchestrator: Orchestrator,
    sources: Sequence[TriggerSource],
    *,
    now: str | datetime | None = None,
) -> ScanOutcome:
    """Ask every source for triggers and enqueue the new ones.

    Sources are scanned in the order given and acknowledged one by one, so a
    crash mid-way leaves only the unscanned sources for the next pass. Every
    trigger goes through the idempotent ``enqueue``: a duplicate is counted, not
    duplicated, which is what makes the two-phase protocol safe.
    """
    queued: list[Job] = []
    duplicates: list[Job] = []
    names: list[str] = []
    for source in sources:
        names.append(source.name)
        triggers = list(source.scan())
        for trigger in triggers:
            job, created = orchestrator.store.enqueue(
                trigger.kind,
                dedupe_key=trigger.dedupe_key,
                payload=trigger.payload,
                priority=trigger.priority,
                max_retries=trigger.max_retries,
                depends_on=trigger.depends_on,
                now=now,
            )
            (queued if created else duplicates).append(job)
        source.seen(triggers)
    orchestrator.store.append_event(
        "trigger_scan",
        detail=json.dumps(
            {
                "sources": names,
                "queued": len(queued),
                "duplicates": len(duplicates),
            },
            sort_keys=True,
            separators=(",", ":"),
        ),
        now=now,
    )
    return ScanOutcome(sources=tuple(names), queued=tuple(queued), duplicates=tuple(duplicates))


class FileTriggerSource:
    """New or changed files under one directory, identified by content.

    A file is skipped without hashing when its size and ``mtime_ns`` match the
    cursor entry; otherwise its SHA-256 is computed and becomes the trigger's
    ``dedupe_key``. Content identity means a byte-identical copy under another
    name does not queue the work twice, and an unrelated edit does.

    The cursor lives in the job store (one JSON document per source) rather than
    a sidecar file, so the queue and its cursors survive or are lost together.
    Renamed files appear as one new trigger; removed files drop out of the
    cursor. Hashing is O(size) once per change, which is acceptable for the
    data/document folders this is meant for and unacceptable for a directory of
    multi-gigabyte files — that is a caller's choice, not a hidden default.
    """

    def __init__(
        self,
        directory: str | Path,
        *,
        kind: str,
        store: JobStore,
        pattern: str = "**/*",
        priority: int = 0,
        max_retries: int = 0,
        name: str | None = None,
    ) -> None:
        self.directory = Path(directory)
        if not self.directory.is_dir():
            raise NotADirectoryError(f"{self.directory} is not a directory")
        if not kind.strip():
            raise ValueError("kind must not be empty")
        self.kind = kind
        self.store = store
        self.pattern = pattern
        self.priority = priority
        self.max_retries = max_retries
        self.name = name or f"file:{self.directory}"
        self.cursor_key = f"trigger.file:{kind}:{self.directory.resolve()}:{pattern}"
        self._pending: dict[str, dict[str, object]] | None = None

    def scan(self) -> list[Trigger]:
        """Files whose content the cursor has not recorded yet."""
        cursor = self._load_cursor()
        pending: dict[str, dict[str, object]] = {}
        triggers: list[Trigger] = []
        for path in self._list_files():
            try:
                stat = path.stat()
            except FileNotFoundError:
                continue  # removed between listing and stat
            relative = path.relative_to(self.directory).as_posix()
            entry = cursor.get(relative)
            size = int(stat.st_size)
            mtime_ns = int(stat.st_mtime_ns)
            if (
                isinstance(entry, Mapping)
                and entry.get("size") == size
                and entry.get("mtime_ns") == mtime_ns
            ):
                pending[relative] = dict(entry)
                continue
            digest = _sha256(path)
            pending[relative] = {"size": size, "mtime_ns": mtime_ns, "sha256": digest}
            triggers.append(
                Trigger(
                    kind=self.kind,
                    dedupe_key=f"{self.kind}:file:{digest}",
                    payload={
                        "path": str(path),
                        "relative_path": relative,
                        "size": size,
                        "mtime_ns": mtime_ns,
                        "sha256": digest,
                    },
                    priority=self.priority,
                    max_retries=self.max_retries,
                )
            )
        self._pending = pending
        return triggers

    def seen(self, triggers: Sequence[Trigger]) -> None:
        """Persist the cursor built by the last :meth:`scan`.

        Called after the orchestrator enqueued the triggers. Passing the
        triggers themselves is not required to save the cursor, but the argument
        keeps the protocol uniform and lets a future source record per-trigger
        acknowledgements.
        """
        if self._pending is None:
            return
        self.store.set_state(
            self.cursor_key,
            json.dumps({"files": self._pending}, sort_keys=True, separators=(",", ":")),
        )
        self._pending = None

    def _list_files(self) -> list[Path]:
        return sorted(
            (path for path in self.directory.glob(self.pattern) if path.is_file()),
            key=lambda path: path.relative_to(self.directory).as_posix(),
        )

    def _load_cursor(self) -> dict[str, Mapping[str, object]]:
        raw = self.store.get_state(self.cursor_key)
        if raw is None:
            return {}
        try:
            parsed = json.loads(raw)
        except ValueError:
            return {}  # a corrupt cursor only costs a re-hash; dedupe still holds
        if not isinstance(parsed, dict):
            return {}
        files = parsed.get("files")
        if not isinstance(files, dict):
            return {}
        return {str(name): entry for name, entry in files.items() if isinstance(entry, Mapping)}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(_CHUNK_BYTES)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()
