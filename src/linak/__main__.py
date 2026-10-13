import sys

from linak.gui.app import main as gui_main
from linak.cli import main as cli_main


if __name__ == "__main__":
    if "--pipeline" in sys.argv:
        raise SystemExit(cli_main(sys.argv[1:]))
    raise SystemExit(gui_main())
