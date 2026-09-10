# config.py - Cache_Service Configuration
from dataclasses import dataclass


@dataclass
class ServiceConfig:
    service_name: str = 'cache_service'
    buffer_size: int = 1024
    debug_mode: bool = False
