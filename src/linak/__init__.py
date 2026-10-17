"""LiNaK application package."""

from importlib.metadata import PackageNotFoundError, version

try:
	__version__ = version("LiNak")
except PackageNotFoundError:
	__version__ = "0+unknown"
