# main.py - Task Manager Entry Point
import sys
from storage import TaskStorage
from cli import TaskCLI


def main():
    storage = TaskStorage()
    t1 = storage.add_task('Init architecture', priority='HIGH')
    t2 = storage.add_task('Run tests', priority='CRITICAL')
    t2.mark_completed()

    if '--verify' in sys.argv:
        assert len(storage.list_tasks()) == 2
        assert len(storage.list_tasks(completed=True)) == 1
        print('TASK MANAGER VERIFICATION OK')
        return 0

    print(TaskCLI.format_table(storage.list_tasks()))
    return 0


if __name__ == '__main__':
    sys.exit(main())
