# models.py - Task Data Models
import time
from dataclasses import dataclass, field
from config import TaskConfig


@dataclass
class TaskItem:
    id: int
    title: str
    priority: str = 'MEDIUM'
    completed: bool = False
    created_at: float = field(default_factory=time.time)

    def mark_completed(self) -> None:
        self.completed = True

    def update_priority(self, new_priority: str) -> None:
        p_upper = new_priority.upper()
        if p_upper in TaskConfig.valid_priorities:
            self.priority = p_upper
        else:
            raise ValueError(f'Invalid priority: {new_priority}')
