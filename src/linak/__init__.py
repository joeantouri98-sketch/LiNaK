"""LiNaK application package."""

import os as _os

# One BLAS thread per process unless the user sets a count. LiNaK's matrices have a few hundred rows,
# where threaded BLAS gains nothing (the dhf stage runs as fast on 1 thread as on the 4-6 it takes by
# default) while its spinning threads slow to a crawl when the CPU is shared: stages beside the GUI,
# a test suite, a second run. Set before numpy is imported; stage subprocesses inherit it.
for _var in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
	_os.environ.setdefault(_var, "1")
from importlib.metadata import PackageNotFoundError, version

try:
	__version__ = version("LiNak")
except PackageNotFoundError:
	__version__ = "0+unknown"
