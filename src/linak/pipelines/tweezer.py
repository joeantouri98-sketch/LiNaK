"""Optical tweezer dataset pipeline."""
from .runner import run

def main(argv=None):
    return run("tweezer", argv)

if __name__ == "__main__":
    raise SystemExit(main())
