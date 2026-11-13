"""Old name of the ``qali`` package (LiNaK was renamed Qali in 2026-11).

``import linak`` and ``import linak.<anything>`` still work and give the very same
modules as ``qali``; new code should import ``qali``.
"""

import importlib
import importlib.abc
import importlib.util
import sys
import warnings

import qali

warnings.warn("the 'linak' package is now 'qali'; import qali instead", DeprecationWarning, stacklevel=2)


class _Alias(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Resolve ``linak.x.y`` to the module ``qali.x.y`` itself."""

    def find_spec(self, name, path=None, target=None):
        if name.startswith("linak."):
            return importlib.util.spec_from_loader(name, self)
        return None

    def create_module(self, spec):
        return importlib.import_module("qali" + spec.name[len("linak"):])

    def exec_module(self, module):
        pass


sys.meta_path.insert(0, _Alias())
sys.modules[__name__] = qali
