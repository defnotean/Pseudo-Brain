# tests/test_pipeline.py - Multi-File Math Tests
import sys
from pathlib import Path
root = Path(__file__).parent.parent.resolve()
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from matrix import Matrix
from stats import matrix_mean, matrix_std


def test_matrix_operations():
    m1 = Matrix(rows=2, cols=2, data=[[1.0, 2.0], [3.0, 4.0]])
    t = m1.transpose()
    assert t.data == [[1.0, 3.0], [2.0, 4.0]]
    dot = m1.dot(t)
    assert dot.rows == 2 and dot.cols == 2
    assert dot.data[0][0] == 5.0


def test_stats():
    m = Matrix(rows=2, cols=2, data=[[2.0, 4.0], [4.0, 6.0]])
    assert matrix_mean(m) == 4.0


if __name__ == '__main__':
    test_matrix_operations()
    test_stats()
    print('MATH PIPELINE MULTI-FILE TESTS PASSED (2/2)!')
