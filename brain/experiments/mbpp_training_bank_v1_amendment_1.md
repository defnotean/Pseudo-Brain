# MBPP training-bank v1: docstring normalization amendment

The r0 read-only diagnostic proved that adding a module-level docstring changed
the structural code fingerprint, despite its specified removal. Declaration
identifiers were numbered by positions in the entire module before docstring
removal. The pinned 974 MBPP source rows contain no module docstrings, but the
protected development inputs also participate in isolation.

Number only top-level function/class declarations, in their relative order.
Retain all other AST and input normalization, candidate/control selection,
limits, reference programs, tests, and tokenization unchanged. Add a regression
check for equality with and without module documentation. Preserve r0's source,
actual controls, and diagnosis; mark it superseded for downstream training.

Freeze a new r1 source snapshot and rerun the original 374-candidate audit from
scratch, including all applicable positive/negative controls and original
900-second limit. Do not silently carry r0 eligibility into r1. No model
training or capability evaluation is authorized by this amendment.
