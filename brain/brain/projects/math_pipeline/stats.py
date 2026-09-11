# stats.py - Statistical Transformations
import math
from typing import List
from matrix import Matrix


def matrix_mean(m: Matrix) -> float:
    total = sum(val for row in m.data for val in row)
    return total / (m.rows * m.cols)


def matrix_std(m: Matrix) -> float:
    mean = matrix_mean(m)
    variance = sum((val - mean) ** 2 for row in m.data for val in row) / (m.rows * m.cols)
    return math.sqrt(variance)
