# main.py - CLI Entry Point
import sys
from matrix import Matrix
from stats import matrix_mean, matrix_std


def run_pipeline():
    m1 = Matrix(rows=2, cols=3, data=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    m2 = Matrix(rows=3, cols=2, data=[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    res = m1.dot(m2)
    mean_val = matrix_mean(res)
    std_val = matrix_std(res)
    print(f'Matrix Dot Product Output Shape: ({res.rows}, {res.cols})')
    print(f'Mean: {mean_val:.4f}, Std: {std_val:.4f}')
    return True


if __name__ == '__main__':
    run_pipeline()
