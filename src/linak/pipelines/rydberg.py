"""Rydberg dataset generation pipeline."""
from .runner import run

def main(argv=None):
    return run("rydberg", argv)

if __name__ == "__main__":
    raise SystemExit(main())
