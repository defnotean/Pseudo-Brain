# main.py - Cache_Service Entrypoint
import sys
from config import ServiceConfig
from core import ServiceEngine
from interface import ServiceFormatter


def main():
    cfg = ServiceConfig()
    engine = ServiceEngine(cfg)
    engine.put('init_status', 'SUCCESS')
    if '--verify' in sys.argv:
        assert engine.count() == 1
        assert engine.get('init_status').value == 'SUCCESS'
        print('GENERAL UTILITY VERIFICATION OK')
        return 0
    print(ServiceFormatter.summarize(engine))
    return 0


if __name__ == '__main__':
    sys.exit(main())
