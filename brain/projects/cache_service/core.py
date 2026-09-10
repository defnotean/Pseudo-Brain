# core.py - Cache_Service Core Processing Engine
from typing import Any, Dict, List, Optional
from config import ServiceConfig
from models import DataRecord


class ServiceEngine:
    def __init__(self, config: ServiceConfig = None):
        self.cfg = config or ServiceConfig()
        self._records: Dict[str, DataRecord] = {}

    def put(self, key: str, value: Any) -> DataRecord:
        rec = DataRecord(key=key, value=value)
        self._records[key] = rec
        return rec

    def get(self, key: str) -> Optional[DataRecord]:
        return self._records.get(key)

    def count(self) -> int:
        return len(self._records)
