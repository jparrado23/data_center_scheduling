import pytest

gp = pytest.importorskip("gurobipy")

from src.data.synthetic import generate_toy_dataset
from src.evaluation.results import extract_schedule
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
