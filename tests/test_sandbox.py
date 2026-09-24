"""Minimal tests for the enforcement boundary.

Run: pytest -q. These check the sandbox rejects unsafe code and runs safe code.
The pattern transforms need a Spark session, so they're covered separately in a
Databricks job test, not here.
"""
from __future__ import annotations

# pytest idioms: fixtures are injected by argument name; test names document intent.
# pylint: disable=missing-function-docstring

import pandas as pd
import pytest

from src.execution.sandbox import UnsafeCodeError, execute, static_check


def test_rejects_import():
    with pytest.raises(UnsafeCodeError):
        static_check("import os\nresult = 1")


def test_rejects_dunder():
    with pytest.raises(UnsafeCodeError):
        static_check("result = ().__class__.__bases__")


def test_rejects_forbidden_name():
    with pytest.raises(UnsafeCodeError):
        static_check("result = eval('1+1')")


def test_runs_safe_dataframe_code():
    frames = {"tx": pd.DataFrame({"party_id": ["a", "a", "b"], "amount_base": [10, 20, 5]})}
    code = "result = tx.groupby('party_id')['amount_base'].sum().reset_index()"
    res = execute(code, frames, timeout_s=15)
    assert res.ok and res.kind == "dataframe"
    assert len(res.payload) == 2


def test_requires_result_assignment():
    frames = {"tx": pd.DataFrame({"amount_base": [1, 2]})}
    res = execute("x = tx.sum()", frames, timeout_s=15)
    assert not res.ok and res.kind == "error"
