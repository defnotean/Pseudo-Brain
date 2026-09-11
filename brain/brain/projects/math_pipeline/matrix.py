# matrix.py - Core Matrix Data Structure
from dataclasses import dataclass
from typing import List


@dataclass
class Matrix:
    rows: int
    cols: int
    data: List[List[float]]

    def transpose(self) -> 'Matrix':
        t_data = [[self.data[r][c] for r in range(self.rows)] for c in range(self.cols)]
        return Matrix(rows=self.cols, cols=self.rows, data=t_data)

    def dot(self, other: 'Matrix') -> 'Matrix':
        assert self.cols == other.rows, f'Dimension mismatch: {self.cols} != {other.rows}'
        out_data = [
            [sum(self.data[r][k] * other.data[k][c] for k in range(self.cols)) for c in range(other.cols)]
            for r in range(self.rows)
        ]
        return Matrix(rows=self.rows, cols=other.cols, data=out_data)
