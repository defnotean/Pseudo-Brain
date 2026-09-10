# tests/test_tasks.py - Unit Tests for Task Manager
from config import TaskConfig
from models import TaskItem
from storage import TaskStorage
from cli import TaskCLI


def test_task_creation_and_completion():
    storage = TaskStorage()
    t = storage.add_task('Test task', priority='HIGH')
    assert t.id == 1
    assert t.completed is False
    t.mark_completed()
    assert t.completed is True


def test_task_filtering_and_removal():
    storage = TaskStorage()
    storage.add_task('Task 1')
    t2 = storage.add_task('Task 2')
    t2.mark_completed()
    assert len(storage.list_tasks(completed=True)) == 1
    assert len(storage.list_tasks(completed=False)) == 1
    storage.remove_task(1)
    assert len(storage.list_tasks()) == 1


def test_cli_table_formatting():
    storage = TaskStorage()
    t = storage.add_task('Sample item')
    output = TaskCLI.format_table([t])
    assert 'Sample item' in output
    assert 'OPEN' in output


if __name__ == '__main__':
    test_task_creation_and_completion()
    test_task_filtering_and_removal()
    test_cli_table_formatting()
    print('TASK MANAGER MULTI-FILE TESTS PASSED CLEANLY (3/3)!')
