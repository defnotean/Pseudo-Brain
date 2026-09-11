# config.py - Task Manager Configuration
from dataclasses import dataclass


@dataclass
class TaskConfig:
    storage_file: str = 'tasks.json'
    default_priority: str = 'MEDIUM'
    valid_priorities: tuple = ('LOW', 'MEDIUM', 'HIGH', 'CRITICAL')
