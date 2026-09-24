"""Sandboxed execution of agent-generated code.

The cash agent executed generated Python in a sandboxed namespace and returned a
DataFrame or a Plotly figure. This is the same idea, hardened. Prompt guardrails
are advisory; THIS module is the enforcement boundary:

  * a restricted builtins set (no open/eval/exec/import/__import__),
  * a curated, read-only namespace (pandas frames + px/pd/np already bound),
  * an AST pre-check that rejects imports, attribute access to dunders, and
    calls to obviously dangerous names,
  * a wall-clock timeout,
  * the convention that the result must be assigned to `result`.

This is defence-in-depth, not a perfect jail. For untrusted input at scale, run
the executor in a separate process / container with no network and a CPU+memory
cgroup — the interface below stays the same.
"""
from __future__ import annotations

import ast
import multiprocessing as mp
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px

_FORBIDDEN_NAMES = {
    "eval", "exec", "compile", "open", "__import__", "input",
    "globals", "locals", "vars", "getattr", "setattr", "delattr",
    "exit", "quit", "help", "memoryview",
}

_SAFE_BUILTINS = {
    k: __builtins__[k] if isinstance(__builtins__, dict) else getattr(__builtins__, k)
    for k in (
        "abs", "min", "max", "sum", "len", "range", "enumerate", "zip", "map",
        "filter", "sorted", "round", "float", "int", "str", "bool", "list",
        "dict", "set", "tuple", "any", "all", "isinstance",
    )
}


@dataclass
class ExecResult:
    ok: bool
    kind: str          # "dataframe" | "figure" | "error"
    payload: Any       # DataFrame | figure JSON | error string
    stdout: str = ""


class UnsafeCodeError(Exception):
    pass


def static_check(code: str) -> None:
    """Reject code that imports, touches dunders, or calls forbidden names."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        raise UnsafeCodeError(f"Syntax error: {e}") from e

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            raise UnsafeCodeError("Imports are not allowed.")
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            raise UnsafeCodeError(f"Access to dunder attribute '{node.attr}' is blocked.")
        if isinstance(node, ast.Name) and node.id in _FORBIDDEN_NAMES:
            raise UnsafeCodeError(f"Use of '{node.id}' is blocked.")


def _run(code: str, frames: dict[str, pd.DataFrame], q: "mp.Queue") -> None:
    import io
    import contextlib

    ns: dict[str, Any] = {
        "__builtins__": _SAFE_BUILTINS,
        "pd": pd, "np": np, "px": px,
        **frames,
        "result": None,
    }
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            exec(code, ns)  # noqa: S102 — constrained namespace + AST pre-check
        result = ns.get("result")
        if isinstance(result, pd.DataFrame):
            q.put(ExecResult(True, "dataframe", result.head(1000).to_dict("records"), buf.getvalue()))
        elif hasattr(result, "to_json"):        # Plotly figure
            q.put(ExecResult(True, "figure", result.to_json(), buf.getvalue()))
        else:
            q.put(ExecResult(False, "error", "Code did not assign a DataFrame or figure to `result`.", buf.getvalue()))
    except Exception as e:  # noqa: BLE001 — surface any runtime error to the caller
        q.put(ExecResult(False, "error", f"{type(e).__name__}: {e}", buf.getvalue()))


def execute(code: str, frames: dict[str, pd.DataFrame], timeout_s: int = 30) -> ExecResult:
    """Static-check then run the code in a separate process with a timeout."""
    static_check(code)

    q: "mp.Queue" = mp.Queue()
    proc = mp.Process(target=_run, args=(code, frames, q))
    proc.start()
    proc.join(timeout_s)
    if proc.is_alive():
        proc.terminate()
        proc.join()
        return ExecResult(False, "error", f"Execution exceeded {timeout_s}s and was terminated.")
    return q.get() if not q.empty() else ExecResult(False, "error", "No result returned.")
