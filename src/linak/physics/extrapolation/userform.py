"""A user-written core polarization correction g(r), evaluated safely.

The dipole operator is r * [1 - lam * s(w) * g(r)]. The formula is parsed with ``ast`` and evaluated by a small
interpreter that only knows numbers, the variables below, the user's own parameters and a short list of math functions;
nothing else (attributes, indexing, calls to anything else) is accepted. A parameter is either a constant or a table of
values against r (imported from a dataset) that is interpolated onto the radial grid.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import numpy as np

MAX_FORMULA = 500
_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

FUNCTIONS = {
    "exp": np.exp, "sqrt": np.sqrt, "log": np.log, "log10": np.log10, "sin": np.sin, "cos": np.cos, "tan": np.tan,
    "tanh": np.tanh, "sinh": np.sinh, "cosh": np.cosh, "arctan": np.arctan, "abs": np.abs,
}
VARIABLES = ("r", "alpha_D", "rc", "pi", "e")
# Names the model already uses (settings and fields): a user parameter may not reuse any of them.
_OUR_NAMES = ("lam", "wcore_eV", "so_scale", "singor_alpha_D", "singor_rc", "form", "nl", "np", "numpy")
RESERVED = {n.lower() for n in (*VARIABLES, *FUNCTIONS, *_OUR_NAMES)}


class FormulaError(ValueError):
    """The formula or a parameter is not valid; the message says what to fix."""


def check_name(name: str, taken: set[str] | None = None) -> None:
    if not _NAME.match(name or ""):
        raise FormulaError(f"'{name}' is not a valid name (letters, digits and underscore; it cannot start with a digit)")
    if name.lower() in RESERVED:
        raise FormulaError(f"'{name}' is already used by the model; choose another name")
    if taken and name.lower() in taken:
        raise FormulaError(f"'{name}' is used twice")


def _param_table(p: dict) -> tuple[np.ndarray, np.ndarray] | None:
    t = p.get("table")
    if not t:
        return None
    return np.asarray(t["r"], float), np.asarray(t["y"], float)


def _bind(params: list[dict], r, alpha_d: float, rc: float) -> dict:
    env = {"r": r, "alpha_D": alpha_d, "rc": rc, "pi": np.pi, "e": np.e}
    for p in params:
        tab = _param_table(p)
        env[p["name"]] = np.interp(r, *tab) if tab is not None else float(p["value"])   # edge values are held outside the table
    return env


def _eval(node, env):
    if isinstance(node, ast.Expression):
        return _eval(node.body, env)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return float(node.value)
    if isinstance(node, ast.Name):
        if node.id not in env:
            raise FormulaError(f"unknown name '{node.id}'")
        return env[node.id]
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        v = _eval(node.operand, env)
        return -v if isinstance(node.op, ast.USub) else v
    if isinstance(node, ast.BinOp):
        a, b = _eval(node.left, env), _eval(node.right, env)
        op = node.op
        if isinstance(op, ast.Add):
            return a + b
        if isinstance(op, ast.Sub):
            return a - b
        if isinstance(op, ast.Mult):
            return a * b
        if isinstance(op, ast.Div):
            return a / b
        if isinstance(op, ast.Pow):
            return np.power(a, b)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in FUNCTIONS and not node.keywords:
        if len(node.args) != 1:
            raise FormulaError(f"{node.func.id}() takes one argument")
        return FUNCTIONS[node.func.id](_eval(node.args[0], env))
    raise FormulaError("only numbers, names, + - * / ** , parentheses and the listed functions are allowed")


def evaluate(formula: str, params: list[dict], r: np.ndarray, alpha_d: float, rc: float) -> np.ndarray:
    """g(r) on the grid ``r`` (always an array of the same shape)."""
    if not formula or len(formula) > MAX_FORMULA:
        raise FormulaError("the formula is empty or too long")
    try:
        tree = ast.parse(formula.strip(), mode="eval")
    except SyntaxError as exc:
        raise FormulaError(f"cannot read the formula ({exc.msg})") from exc
    with np.errstate(all="ignore"):
        out = _eval(tree, _bind(params, r, alpha_d, rc))
    return np.broadcast_to(np.asarray(out, float), r.shape).copy()


def validate_user(formula: str, params: list[dict]) -> None:
    """Raise :class:`FormulaError` unless the names are fine and the formula gives finite numbers on a test grid."""
    taken: set[str] = set()
    for p in params:
        check_name(p.get("name", ""), taken)
        taken.add(p["name"].lower())
        tab = _param_table(p)
        if tab is not None:
            if len(tab[0]) < 2 or not np.all(np.diff(tab[0]) > 0) or not np.all(np.isfinite(tab[1])):
                raise FormulaError(f"the dataset of '{p['name']}' needs at least two points with increasing r and finite values")
        else:
            try:
                float(p["value"])
            except (KeyError, TypeError, ValueError) as exc:
                raise FormulaError(f"'{p['name']}' needs a number") from exc
    grid = np.geomspace(1e-3, 100.0, 300)
    g = evaluate(formula, params, grid, 1.0, 2.0)
    if not np.all(np.isfinite(g)):
        raise FormulaError("the formula gives values that are not finite somewhere between r = 0.001 and 100 a0")


def parse_dataset(path: str | Path) -> dict:
    """Read two columns (r in a0, value) from a csv/txt file; header lines are skipped."""
    rows = []
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        parts = [p for p in re.split(r"[,;\s]+", line.strip()) if p]
        try:
            rows.append((float(parts[0]), float(parts[1])))
        except (IndexError, ValueError):
            continue
    if len(rows) < 2:
        raise FormulaError("the file needs at least two lines with two numbers each (r in a0, value)")
    rows.sort()
    r, y = np.array(rows).T
    if not np.all(np.diff(r) > 0):
        raise FormulaError("the r column must not repeat values")
    if not (np.all(np.isfinite(r)) and np.all(np.isfinite(y))):
        raise FormulaError("the file contains values that are not finite")
    return {"r": r.tolist(), "y": y.tolist()}
