"""Durable, profile-scoped outbox for native-memory mirror events."""

from __future__ import annotations

import json
import os
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple


class MirrorOutboxFull(RuntimeError):
    """Raised when the bounded outbox cannot accept another event."""


@dataclass(frozen=True)
class MirrorOutboxItem:
    id: str
    action: str
    target: str
    content: str
    old_content: str
    trusted: Dict[str, str]
    attempt_count: int
    created_at: float


class MirrorOutbox:
    """Small SQLite queue whose rows survive process and device restarts.

    A row stays ``pending`` while it is delivered. If the process dies after
    local-server commits but before Hermes deletes the row, replay uses the
    same trusted tool_call_id and local-server's candidate-key uniqueness makes
    the write idempotent.
    """

    def __init__(self, path: Path, *, max_items: int) -> None:
        self.path = Path(path)
        self.max_items = max(1, int(max_items))
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.path.parent, 0o700)
        if not self.path.exists():
            descriptor = os.open(
                self.path,
                os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                0o600,
            )
            os.close(descriptor)
        os.chmod(self.path, 0o600)
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS mirror_outbox (
                    id TEXT PRIMARY KEY,
                    action TEXT NOT NULL,
                    target TEXT NOT NULL,
                    content TEXT NOT NULL,
                    old_content TEXT NOT NULL,
                    trusted_json TEXT NOT NULL,
                    state TEXT NOT NULL DEFAULT 'pending'
                        CHECK (state IN ('pending', 'dead')),
                    attempt_count INTEGER NOT NULL DEFAULT 0,
                    next_attempt_at REAL NOT NULL,
                    last_error_type TEXT NOT NULL DEFAULT '',
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS mirror_outbox_due
                    ON mirror_outbox(state, next_attempt_at, created_at);
                PRAGMA user_version = 1;
                """
            )
        self._secure_sidecars()

    def enqueue(
        self,
        *,
        item_id: str,
        action: str,
        target: str,
        content: str,
        old_content: str,
        trusted: Dict[str, str],
        now: Optional[float] = None,
    ) -> None:
        timestamp = time.time() if now is None else float(now)
        trusted_json = json.dumps(
            trusted,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            count = int(
                connection.execute("SELECT COUNT(*) FROM mirror_outbox").fetchone()[0]
            )
            if count >= self.max_items:
                connection.rollback()
                raise MirrorOutboxFull(
                    f"deep memory mirror outbox is full ({self.max_items})"
                )
            connection.execute(
                """
                INSERT INTO mirror_outbox (
                    id, action, target, content, old_content, trusted_json,
                    state, attempt_count, next_attempt_at, last_error_type,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, 'pending', 0, ?, '', ?, ?)
                """,
                (
                    item_id,
                    action,
                    target,
                    content,
                    old_content,
                    trusted_json,
                    timestamp,
                    timestamp,
                    timestamp,
                ),
            )
            connection.commit()
        self._secure_sidecars()

    def next_due(
        self, *, now: Optional[float] = None
    ) -> Tuple[Optional[MirrorOutboxItem], Optional[float]]:
        timestamp = time.time() if now is None else float(now)
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT id, action, target, content, old_content, trusted_json,
                       attempt_count, next_attempt_at, created_at
                FROM mirror_outbox
                WHERE state = 'pending'
                ORDER BY next_attempt_at ASC, created_at ASC
                LIMIT 1
                """
            ).fetchone()
        if row is None:
            return None, None
        delay = max(0.0, float(row[7]) - timestamp)
        if delay > 0:
            return None, delay
        trusted = json.loads(str(row[5]))
        if not isinstance(trusted, dict) or not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in trusted.items()
        ):
            raise ValueError("deep memory mirror outbox metadata is invalid")
        return MirrorOutboxItem(
            id=str(row[0]),
            action=str(row[1]),
            target=str(row[2]),
            content=str(row[3]),
            old_content=str(row[4]),
            trusted=trusted,
            attempt_count=int(row[6]),
            created_at=float(row[8]),
        ), 0.0

    def complete(self, item_id: str) -> None:
        with self._connect() as connection:
            connection.execute("DELETE FROM mirror_outbox WHERE id = ?", (item_id,))

    def retry(
        self,
        item_id: str,
        *,
        error_type: str,
        delay: float,
        now: Optional[float] = None,
    ) -> None:
        timestamp = time.time() if now is None else float(now)
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE mirror_outbox
                SET state = 'pending',
                    attempt_count = attempt_count + 1,
                    next_attempt_at = ?,
                    last_error_type = ?,
                    updated_at = ?
                WHERE id = ?
                """,
                (
                    timestamp + max(0.0, float(delay)),
                    str(error_type)[:32],
                    timestamp,
                    item_id,
                ),
            )

    def mark_dead(
        self,
        item_id: str,
        *,
        error_type: str,
        now: Optional[float] = None,
    ) -> None:
        timestamp = time.time() if now is None else float(now)
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE mirror_outbox
                SET state = 'dead',
                    attempt_count = attempt_count + 1,
                    last_error_type = ?,
                    updated_at = ?
                WHERE id = ?
                """,
                (str(error_type)[:32], timestamp, item_id),
            )

    def counts(self) -> Dict[str, int]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT state, COUNT(*) FROM mirror_outbox GROUP BY state"
            ).fetchall()
        counts = {"pending": 0, "dead": 0}
        counts.update({str(state): int(count) for state, count in rows})
        return counts

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self.path), timeout=5.0)
        connection.execute("PRAGMA busy_timeout = 5000")
        # The managed device currently links SQLite 3.40.x, which is affected
        # by the upstream WAL-reset corruption bug. This queue has one drain
        # worker and only brief hook-side inserts, so rollback journaling gives
        # us the required crash durability without relying on vulnerable WAL.
        connection.execute("PRAGMA journal_mode = DELETE")
        connection.execute("PRAGMA synchronous = FULL")
        return connection

    def _secure_sidecars(self) -> None:
        for candidate in (
            self.path,
            Path(f"{self.path}-wal"),
            Path(f"{self.path}-shm"),
        ):
            if candidate.exists():
                os.chmod(candidate, 0o600)
