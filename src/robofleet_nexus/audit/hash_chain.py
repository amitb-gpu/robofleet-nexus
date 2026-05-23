from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field


class AuditRecord(BaseModel):
    sequence: int
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
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
