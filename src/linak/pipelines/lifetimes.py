"""Radiative lifetime dataset pipeline."""
from .runner import run

def main(argv=None):
    return run("lifetimes", argv)

if __name__ == "__main__":
    raise SystemExit(main())
