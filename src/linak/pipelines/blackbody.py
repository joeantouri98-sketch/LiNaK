"""Blackbody radiation dataset pipeline."""
from .runner import run

def main(argv=None):
    return run("blackbody", argv)

if __name__ == "__main__":
    raise SystemExit(main())
