"""Hand-computed checks for the structural hardness predictors."""

import numpy as np
import pandas as pd
import pytest

from src.instance_generator.hardness_metrics import compute_hardness_metrics, lp_root_gap


def _toy_instance():
    """Three jobs on a 10-slot horizon with capacity 12 node units.

    Windows (A_i = [e, l + p)):
      job a: d=4, p=2, starts 0..2 -> active slots 0..3
      job b: d=6, p=3, starts 2..4 -> active slots 2..6
      job c: d=8, p=1, starts 3..3 -> active slot 3
    """

    jobs = pd.DataFrame(
        {
            "job_id": ["a", "b", "c"],
            "node_units_required": [4, 6, 8],
            "duration": [2, 3, 1],
            "earliest_start": [0, 2, 3],
            "latest_start": [2, 4, 3],
            "power": [1.0, 1.0, 1.0],
            "compatible_clusters": ["k", "k", "k"],
        }
    )
    price = [5.0, 4.0, 3.0, 1.0, 2.0, 2.0, 6.0, 7.0, 8.0, 9.0]
    energy = pd.DataFrame({"hour": range(10), "effective_price": price, "pue": 1.0})
    clusters = pd.DataFrame({"cluster_id": ["k"], "gpu_count": [12]})
    return jobs, energy, clusters


def test_toy_metrics_match_hand_computation() -> None:
    jobs, energy, clusters = _toy_instance()
    metrics = compute_hardness_metrics(jobs, energy, clusters, delta_t=1.0)

    # rho_t = [4, 4, 10, 18, 6, 6, 6, 0, 0, 0] / 12
    assert metrics["max_rho"] == pytest.approx(18 / 12)
    assert metrics["mean_rho"] == pytest.approx(54 / 120)
    assert metrics["slots_rho_gt_1"] == 1
    assert metrics["cheapest_slot"] == 3
    assert metrics["rho_at_cheapest"] == pytest.approx(1.5)
    assert metrics["omega"] == 3
    assert metrics["demand_gcd"] == 2
    assert metrics["dmax_over_c"] == pytest.approx(8 / 12)
    assert metrics["distinct_job_types"] == 3
    assert metrics["utilization"] == pytest.approx((8 + 18 + 8) / 120)
    assert metrics["mean_start_options"] == pytest.approx(7 / 3)
    assert metrics["assignment_variables"] == 7


def test_toy_savings_correlation_and_ties() -> None:
    jobs, energy, clusters = _toy_instance()
    metrics = compute_hardness_metrics(jobs, energy, clusters, delta_t=1.0)

    # a: costs 9, 7, 4 -> saving 5; b: 6, 5, 10 -> 5; c: single start -> 0.
    savings = np.array([5.0, 5.0, 0.0])
    weight = np.array([8.0, 18.0, 8.0])
    assert metrics["savings_weight_corr"] == pytest.approx(np.corrcoef(savings, weight)[0, 1])
    assert metrics["price_tie_share"] == 0.0


def test_tied_cheapest_start_is_detected() -> None:
    jobs, energy, clusters = _toy_instance()
    energy.loc[:, "effective_price"] = 1.0
    metrics = compute_hardness_metrics(jobs, energy, clusters, delta_t=1.0)

    # a and b have several equally cheap starts; c has a single start.
    assert metrics["price_tie_share"] == pytest.approx(2 / 3)


def test_lp_root_gap() -> None:
    jobs, energy, clusters = _toy_instance()
    assert compute_hardness_metrics(jobs, energy, clusters)["lp_root_gap"] is None
    assert lp_root_gap(100.0, 99.5) == pytest.approx(0.005)
    assert lp_root_gap(None, 1.0) is None
