"""Electric-dipole transition dataset pipeline."""
from .runner import run

def main(argv=None):
    return run("transitions", argv)

if __name__ == "__main__":
    raise SystemExit(main())
