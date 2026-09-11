# tests/test_cache_service.py - Cross-Module Verification
import sys
from pathlib import Path
root = Path(__file__).parent.parent.resolve()
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from config import ServiceConfig
from models import DataRecord
from core import ServiceEngine
from interface import ServiceFormatter


def test_service_engine_crud():
    engine = ServiceEngine()
    engine.put('k1', 42)
    assert engine.count() == 1
    rec = engine.get('k1')
    assert rec.value == 42


def test_service_formatter():
    engine = ServiceEngine()
    engine.put('alpha', 'beta')
    summary = ServiceFormatter.summarize(engine)
    assert 'Total Records: 1' in summary


if __name__ == '__main__':
    test_service_engine_crud()
    test_service_formatter()
    print('CACHE_SERVICE MULTI-FILE TESTS PASSED CLEANLY (2/2)!')
