import sys

from linak.cli import main as cli_main


if __name__ == "__main__":
    raise SystemExit(cli_main(sys.argv[1:]))
