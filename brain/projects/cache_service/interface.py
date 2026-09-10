# interface.py - Cache_Service Output Interface
from core import ServiceEngine


class ServiceFormatter:
    @staticmethod
    def summarize(engine: ServiceEngine) -> str:
        return f'Service: {engine.cfg.service_name} | Total Records: {engine.count()}'
