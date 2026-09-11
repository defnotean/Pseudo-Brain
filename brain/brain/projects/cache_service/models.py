# models.py - Cache_Service Domain Models
from dataclasses import dataclass
from typing import Any, Dict
from config import ServiceConfig


@dataclass
class DataRecord:
    key: str
    value: Any
    metadata: Dict[str, Any] = None

    def to_dict(self) -> Dict[str, Any]:
        return {'key': self.key, 'value': self.value, 'metadata': self.metadata or {}}
