"""Solve helpers for Gurobi models."""

from __future__ import annotations

from collections.abc import Callable

import gurobipy as gp
from gurobipy import GRB


def solve_model(
    model: gp.Model,
    time_limit: float | None = None,
    mip_gap: float | None = None,
    *,
    callback: Callable[[gp.Model, int], None] | None = None,
    raise_on_no_solution: bool = True,
) -> gp.Model:
    """Optimize a Gurobi model with optional time and gap limits.

    The model is solved in place. If Gurobi reports an infeasible, unbounded,
    or otherwise unsolved model, the function raises a runtime error with a
    human-readable message instead of returning an unusable result.

    ``callback`` is passed to ``model.optimize`` (for example to record the time
    of the first incumbent). With ``raise_on_no_solution=False`` a time limit
    without an incumbent returns the model so callers can label the outcome;
    infeasible or unbounded models still raise.
    """

    if time_limit is not None:
        model.Params.TimeLimit = time_limit
    if mip_gap is not None:
        model.Params.MIPGap = mip_gap

    if callback is None:
        model.optimize()
    else:
        model.optimize(callback)

    if model.Status in {GRB.INFEASIBLE, GRB.INF_OR_UNBD}:
        raise RuntimeError("MILP model is infeasible or unbounded")
    if model.SolCount == 0 and raise_on_no_solution:
        raise RuntimeError(f"Gurobi finished with status {model.Status} and no feasible solution")
    return model
