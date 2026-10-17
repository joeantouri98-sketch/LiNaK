"""ORCA execution pipeline."""
from .runner import run

def main(argv=None):
    return run("runorca", argv)

if __name__ == "__main__":
    raise SystemExit(main())
