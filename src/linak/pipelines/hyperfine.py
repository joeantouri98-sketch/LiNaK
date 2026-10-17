"""Hyperfine and Zeeman dataset pipeline."""
from .runner import run

def main(argv=None):
    return run("hyperfine", argv)

if __name__ == "__main__":
    raise SystemExit(main())
