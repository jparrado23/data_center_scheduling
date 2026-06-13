import pytest
from dataclasses import replace

gp = pytest.importorskip("gurobipy")

from src.data.synthetic import generate_toy_dataset
from src.evaluation.metrics import compute_summary_metrics
from src.evaluation.results import extract_hourly_results, extract_schedule
from src.milp.gurobi_model import build_milp_model
from src.milp.solve import solve_model


def test_solved_toy_model_assigns_each_job_exactly_once():
    """Verify the toy MILP solution returns one scheduled assignment per job.

    This guards the core assignment constraint and confirms the extraction
    helper can recover a readable schedule from the optimized model state.
    """

    jobs_df, hourly_df, clusters_df, config = generate_toy_dataset()
    model, variables = build_milp_model(jobs_df, hourly_df, clusters_df, config)
    model.Params.OutputFlag = 0

    try:
        solve_model(model)
    except gp.GurobiError as exc:
        pytest.skip(f"Gurobi is installed but not usable in this environment: {exc}")

    schedule = extract_schedule(jobs_df, variables)

    assert len(schedule) == len(jobs_df)
    assert schedule["job_id"].nunique() == len(jobs_df)
    assert sorted(schedule["job_id"].tolist()) == sorted(jobs_df["job_id"].tolist())


def test_solved_toy_model_respects_cluster_compatibility():
    """Verify selected assignments obey job-category compatibility."""

    jobs_df, hourly_df, clusters_df, config = generate_toy_dataset()
    model, variables = build_milp_model(jobs_df, hourly_df, clusters_df, config)
    model.Params.OutputFlag = 0

    try:
        solve_model(model)
    except gp.GurobiError as exc:
        pytest.skip(f"Gurobi is installed but not usable in this environment: {exc}")

    schedule = extract_schedule(jobs_df, variables)
    cluster_categories = {
        row.cluster_id: set(row.compatible_categories) for row in clusters_df.itertuples(index=False)
    }

    for row in schedule.itertuples(index=False):
        assert row.category in cluster_categories[row.assigned_cluster]


def test_solved_toy_model_reports_baseline_and_curtailment():
    """Verify fixed baseline load and renewable curtailment flow to results."""

    jobs_df, hourly_df, clusters_df, config = generate_toy_dataset()
    hourly_df = hourly_df.copy()
    hourly_df["baseline_load"] = 0.002
    model, variables = build_milp_model(jobs_df, hourly_df, clusters_df, config)
    model.Params.OutputFlag = 0

    try:
        solve_model(model)
    except gp.GurobiError as exc:
        pytest.skip(f"Gurobi is installed but not usable in this environment: {exc}")

    hourly_results = extract_hourly_results(hourly_df, variables)

    assert "baseline_load" in hourly_results.columns
    assert "renewable_curtailment" in hourly_results.columns
    assert (hourly_results["baseline_load"] == 0.002).all()
    assert (
        hourly_results["total_load"].round(10)
        == (hourly_results["baseline_load"] + hourly_results["flexible_load"]).round(10)
    ).all()
    assert (
        hourly_results["renewable_curtailment"].round(10)
        == (hourly_results["renewable_available"] - hourly_results["renewable_consumption"]).clip(lower=0).round(10)
    ).all()


def test_solved_toy_model_uses_renewable_before_grid_even_with_negative_prices():
    """The contracted renewable profile is used first, not price-arbitraged away."""

    jobs_df, hourly_df, clusters_df, config = generate_toy_dataset()
    hourly_df = hourly_df.copy()
    hourly_df["grid_price"] = -100.0
    model, variables = build_milp_model(jobs_df, hourly_df, clusters_df, config)
    model.Params.OutputFlag = 0

    try:
        solve_model(model)
    except gp.GurobiError as exc:
        pytest.skip(f"Gurobi is installed but not usable in this environment: {exc}")

    hourly_results = extract_hourly_results(hourly_df, variables)
    expected_renewable = hourly_results[["renewable_available", "total_load"]].min(axis=1)

    assert (hourly_results["renewable_consumption"].round(10) == expected_renewable.round(10)).all()


def test_solved_toy_model_objective_matches_reported_cost_metrics():
    """Keep the Gurobi objective synchronized with post-solve cost reporting."""

    jobs_df, hourly_df, clusters_df, config = generate_toy_dataset()
    model, variables = build_milp_model(jobs_df, hourly_df, clusters_df, config)
    model.Params.OutputFlag = 0

    try:
        solve_model(model)
    except gp.GurobiError as exc:
        pytest.skip(f"Gurobi is installed but not usable in this environment: {exc}")

    hourly_results = extract_hourly_results(hourly_df, variables)
    metrics = compute_summary_metrics(hourly_results, config)

    assert model.ObjVal == pytest.approx(metrics["total_cost"])
    assert metrics["peak_cost"] == pytest.approx(metrics["peak_load"] * config.peak_price)


def test_model_enforces_contracted_power_as_hard_cap():
    """A schedule that cannot fit under contracted power is infeasible."""

    jobs_df, hourly_df, clusters_df, config = generate_toy_dataset()
    config = replace(config, contracted_power=0.001)
    model, _ = build_milp_model(jobs_df, hourly_df, clusters_df, config)
    model.Params.OutputFlag = 0

    try:
        with pytest.raises(RuntimeError, match="infeasible"):
            solve_model(model)
    except gp.GurobiError as exc:
        pytest.skip(f"Gurobi is installed but not usable in this environment: {exc}")
