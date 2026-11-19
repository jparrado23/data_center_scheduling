"""Structural hardness predictors for single-cluster scheduling instances.

All predictors are computed from the instance alone (before solving), except
``lp_root_gap``, which needs the proxy-MILP objective and its LP relaxation.
Notation: job ``i`` has node demand ``d_i``, duration ``p_i`` and start window
``[e_i, l_i]``; it can be active in slots ``A_i = [e_i, l_i + p_i)``.
"""

from __future__ import annotations

from functools import reduce
from math import gcd
from typing import Any

import numpy as np
import pandas as pd


TIE_REL_TOL = 1e-9


def potential_active_matrix(jobs: pd.DataFrame, horizon_slots: int) -> np.ndarray:
    """Return a boolean (jobs x slots) matrix: slot ``t`` lies in ``A_i``."""

    active = np.zeros((len(jobs), horizon_slots), dtype=bool)
    for row, job in enumerate(jobs.itertuples(index=False)):
        first = int(job.earliest_start)
        last = min(horizon_slots, int(job.latest_start) + int(job.duration))
        active[row, first:last] = True
    return active


def job_start_costs(jobs: pd.DataFrame, energy: pd.DataFrame, delta_t: float | None = None) -> list[np.ndarray]:
    """Return the energy cost of every allowed start of every job.

    ``cost(i, s) = power_i * delta_t * sum_{t=s}^{s+p_i-1} PUE_t * price_t``,
    with ``price`` the ``effective_price`` column (falling back to
    ``grid_price``). This is exactly the linear energy coefficient used by the
    proxy MILP and the QUBO.
    """

    ordered = energy.sort_values("hour")
    price_column = "effective_price" if "effective_price" in ordered.columns else "grid_price"
    price = ordered[price_column].to_numpy(dtype=float)
    pue = ordered["pue"].to_numpy(dtype=float) if "pue" in ordered.columns else np.ones(len(price))
    step = 24.0 / len(price) if delta_t is None else float(delta_t)
    weighted = np.concatenate([[0.0], np.cumsum(price * pue)])
    costs = []
    for job in jobs.itertuples(index=False):
        starts = np.arange(int(job.earliest_start), int(job.latest_start) + 1)
        duration = int(job.duration)
        costs.append(float(job.power) * step * (weighted[starts + duration] - weighted[starts]))
    return costs


def _compatible_cluster_count(job: Any, cluster_ids: set[str]) -> int:
    value = getattr(job, "compatible_clusters", None)
    if value is None or pd.isna(value) or not str(value).strip():
        return len(cluster_ids)
    listed = {item.strip() for item in str(value).split("|") if item.strip()}
    return len(listed & cluster_ids)


def lp_root_gap(mip_objective: float | None, lp_objective: float | None) -> float | None:
    """Return ``(MIP - LP) / MIP`` or ``None`` when either value is missing."""

    if mip_objective is None or lp_objective is None or not np.isfinite(mip_objective) or mip_objective == 0:
        return None
    return float((mip_objective - lp_objective) / abs(mip_objective))


def compute_hardness_metrics(
    jobs: pd.DataFrame,
    energy: pd.DataFrame,
    clusters: pd.DataFrame,
    *,
    delta_t: float | None = None,
    proxy_objective: float | None = None,
    lp_objective: float | None = None,
) -> dict[str, Any]:
    """Compute the structural hardness predictors of one instance.

    Capacity ``C`` is the summed ``gpu_count`` of ``clusters`` (node units in
    the single-cluster benchmark). ``lp_root_gap`` is ``None`` unless both the
    proxy-MILP objective and its LP-relaxation objective are supplied.
    """

    horizon = len(energy)
    capacity = int(clusters["gpu_count"].sum())
    demand = jobs["node_units_required"].to_numpy(dtype=int)
    duration = jobs["duration"].to_numpy(dtype=int)
    options = (jobs["latest_start"] - jobs["earliest_start"] + 1).to_numpy(dtype=int)

    active = potential_active_matrix(jobs, horizon)
    rho = (active * demand[:, None]).sum(axis=0) / capacity
    concurrency = active.sum(axis=0)

    ordered = energy.sort_values("hour").reset_index(drop=True)
    price_column = "effective_price" if "effective_price" in ordered.columns else "grid_price"
    cheapest_slot = int(ordered.loc[ordered[price_column].idxmin(), "hour"])

    costs = job_start_costs(jobs, energy, delta_t)
    savings = np.array([cost.max() - cost.min() for cost in costs])
    weight = (demand * duration).astype(float)
    corr = (
        float(np.corrcoef(savings, weight)[0, 1])
        if len(jobs) > 1 and savings.std() > 0 and weight.std() > 0
        else None
    )
    ties = [
        int(np.sum(np.isclose(cost, cost.min(), rtol=TIE_REL_TOL, atol=0.0)) > 1)
        for cost in costs
    ]

    cluster_ids = set(clusters["cluster_id"].astype(str))
    assignment_variables = int(
        sum(
            int(option) * _compatible_cluster_count(job, cluster_ids)
            for option, job in zip(options, jobs.itertuples(index=False), strict=True)
        )
    )

    return {
        "num_jobs": int(len(jobs)),
        "utilization": float((demand * duration).sum() / (capacity * horizon)),
        "max_rho": float(rho.max()),
        "mean_rho": float(rho.mean()),
        "slots_rho_gt_1": int((rho > 1.0 + 1e-12).sum()),
        "cheapest_slot": cheapest_slot,
        "rho_at_cheapest": float(rho[cheapest_slot]),
        "omega": int(concurrency.max()),
        "dmax_over_c": float(demand.max() / capacity),
        "demand_gcd": int(reduce(gcd, [int(value) for value in demand], capacity)),
        "distinct_job_types": int(len(set(zip(demand.tolist(), duration.tolist(), strict=True)))),
        "savings_weight_corr": corr,
        "price_tie_share": float(np.mean(ties)),
        "mean_start_options": float(options.mean()),
        "assignment_variables": assignment_variables,
        "lp_root_gap": lp_root_gap(proxy_objective, lp_objective),
    }
