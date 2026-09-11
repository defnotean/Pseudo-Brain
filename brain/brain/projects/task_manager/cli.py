# cli.py - Formatter and Presentation
from typing import List
from models import TaskItem


class TaskCLI:
    @staticmethod
    def format_table(tasks: List[TaskItem]) -> str:
        if not tasks:
            return 'No tasks available.'
        lines = [f"{'ID':<4} | {'Status':<6} | {'Priority':<8} | {'Title'}"]
        lines.append('-' * 50)
        for t in tasks:
            st = 'DONE' if t.completed else 'OPEN'
            lines.append(f"{t.id:<4} | {st:<6} | {t.priority:<8} | {t.title}")
        return '\n'.join(lines)
