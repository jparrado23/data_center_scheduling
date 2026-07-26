from src.data.synthetic import generate_toy_dataset
from src.data.validation import validate_clusters, validate_hourly_inputs, validate_jobs


def test_generate_toy_dataset_has_expected_schema():
    """Verify the synthetic dataset matches the expected schema and size.

    The test acts as a lightweight guard that the toy data still line up with
    the validators and downstream notebooks after refactoring.
    """

    jobs_df, hourly_df, clusters_df, config = generate_toy_dataset()

    validate_jobs(jobs_df)
    validate_hourly_inputs(hourly_df)
    validate_clusters(clusters_df)

    assert len(hourly_df) == 24
    assert config.contracted_power == 0.20
    assert set(jobs_df.columns) == {"job_id", "category", "duration", "power", "earliest_start", "latest_start"}
    assert set(clusters_df.columns) == {"cluster_id", "capacity"}
    assert set(hourly_df.columns) == {"hour", "renewable_available", "grid_price"}
