"""Run the frozen native restart phase under newly registered strict settings."""
import os

if os.environ.get('CUBLAS_WORKSPACE_CONFIG') != ':4096:8':
    raise RuntimeError('Set registered CUBLAS_WORKSPACE_CONFIG before starting Python')

from run_provenance import apply_deterministic_mode
apply_deterministic_mode()
from check_language_resume_gpu import main

if __name__ == '__main__':
    main()
