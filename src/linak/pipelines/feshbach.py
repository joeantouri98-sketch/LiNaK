"""Feshbach resonance dataset pipeline."""
from .runner import run

def main(argv=None):
    return run("feshbach", argv)

if __name__ == "__main__":
    raise SystemExit(main())
