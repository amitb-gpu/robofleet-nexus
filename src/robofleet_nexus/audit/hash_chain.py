from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field


class AuditRecord(BaseModel):
    sequence: int
    timestamp: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    event_type: str
    actor: str
    payload: dict[str, Any]
    previous_hash: str
    record_hash: str


class HashChainAuditLog:
    def __init__(self) -> None:
        self._records: list[AuditRecord] = []

    def append(self, event_type: str, actor: str, payload: dict[str, Any]) -> AuditRecord:
        previous_hash = self._records[-1].record_hash if self._records else "GENESIS"
        sequence = len(self._records)

        body = {
            "sequence": sequence,
            "event_type": event_type,
            "actor": actor,
            "payload": payload,
            "previous_hash": previous_hash,
        }

        record_hash = hashlib.sha256(
            json.dumps(body, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()

        record = AuditRecord(
            sequence=sequence,
            event_type=event_type,
            actor=actor,
            payload=payload,
            previous_hash=previous_hash,
            record_hash=record_hash,
        )

        self._records.append(record)
        return record

    def records(self) -> list[AuditRecord]:
        return list(self._records)

    @staticmethod
    def verify_records(records: list[AuditRecord]) -> tuple[bool, int | None]:
        """
        Recompute every hash in a chain.

        Returns (True, None) if the chain is intact, otherwise (False, index of the
        first record whose hash or link does not match).
        """
        previous_hash = "GENESIS"
        for index, record in enumerate(records):
            body = {
                "sequence": record.sequence,
                "event_type": record.event_type,
                "actor": record.actor,
                "payload": record.payload,
                "previous_hash": record.previous_hash,
            }
            expected = hashlib.sha256(
                json.dumps(body, sort_keys=True, default=str).encode("utf-8")
            ).hexdigest()
            if (
                record.sequence != index
                or record.previous_hash != previous_hash
                or record.record_hash != expected
            ):
                return False, index
            previous_hash = record.record_hash
        return True, None

    def verify(self) -> tuple[bool, int | None]:
        """Verify this log's chain. See ``verify_records``."""
        return self.verify_records(self._records)
