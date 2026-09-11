# storage.py - In-Memory and Indexed Task Storage
from typing import Dict, List, Optional
from models import TaskItem


class TaskStorage:
    def __init__(self):
        self._tasks: Dict[int, TaskItem] = {}
        self._next_id: int = 1

    def add_task(self, title: str, priority: str = 'MEDIUM') -> TaskItem:
        task = TaskItem(id=self._next_id, title=title, priority=priority)
        self._tasks[self._next_id] = task
        self._next_id += 1
        return task

    def get_task(self, task_id: int) -> Optional[TaskItem]:
        return self._tasks.get(task_id)

    def list_tasks(self, completed: Optional[bool] = None) -> List[TaskItem]:
        if completed is None:
            return list(self._tasks.values())
        return [t for t in self._tasks.values() if t.completed == completed]

    def remove_task(self, task_id: int) -> bool:
        if task_id in self._tasks:
            del self._tasks[task_id]
            return True
        return False
