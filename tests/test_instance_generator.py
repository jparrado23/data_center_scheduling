import pandas as pd

from src.data.validation import validate_clusters, validate_hourly_inputs, validate_jobs
from src.instance_generator import SAMPLING_MODE_PARAMETRIC, SyntheticInstanceConfig, generate_feasible_instance


def _parametric_config(**overrides) -> SyntheticInstanceConfig:
    values = {"num_jobs": 80, "sampling_mode": SAMPLING_MODE_PARAMETRIC}
    values.update(overrides)
    return SyntheticInstanceConfig(**values)


def test_generate_feasible_instance_returns_valid_model_inputs():
    instance = generate_feasible_instance(_parametric_config(seed=7))

    validate_jobs(instance.jobs_df)
    validate_hourly_inputs(instance.hourly_df)
    validate_clusters(instance.clusters_df)

    assert len(instance.jobs_df) == 80
    assert len(instance.hourly_df) == 24
    assert len(instance.hidden_schedule_df) == 80
    assert {"gpu_type_required", "gpu_count_required", "cpu_required", "memory_required_gb"}.issubset(
        instance.jobs_df.columns
    )
    assert set(instance.clusters_df["gpu_type"]) == {"A10", "G2", "G3", "P100", "T4", "V100M16", "V100M32"}
    assert instance.feasibility_report.gpu_utilization_ratio > 0
    assert instance.feasibility_report.power_utilization_ratio > 0


def test_hidden_schedule_obeys_exported_start_windows():
    instance = generate_feasible_instance(
        _parametric_config(
            num_jobs=120,
            seed=11,
            min_window_slack=4,
            max_window_slack=10,
            extra_compatibility_probability=1.0,
        )
    )

    jobs = instance.jobs_df.set_index("job_id")
    for row in instance.hidden_schedule_df.itertuples(index=False):
        job = jobs.loc[row.job_id]
        assert job.earliest_start <= row.start_hour <= job.latest_start
        assert row.end_hour <= len(instance.hourly_df)


def test_hidden_schedule_obeys_cluster_power_and_gpu_capacity():
    instance = generate_feasible_instance(_parametric_config(num_jobs=150, seed=19))

    clusters = instance.clusters_df.set_index("cluster_id")
    horizon = len(instance.hourly_df)
    power_used = {cluster: [0.0] * horizon for cluster in clusters.index}
    gpu_used = {cluster: [0] * horizon for cluster in clusters.index}

    for row in instance.hidden_schedule_df.itertuples(index=False):
        for hour in range(row.start_hour, row.end_hour):
            power_used[row.assigned_cluster][hour] += row.power
            gpu_used[row.assigned_cluster][hour] += row.gpus

    for cluster, power_profile in power_used.items():
        assert max(power_profile) <= clusters.loc[cluster, "capacity"] + 1e-9
        assert max(gpu_used[cluster]) <= clusters.loc[cluster, "gpu_capacity"]


def test_generator_is_reproducible_for_same_seed():
    config = _parametric_config(num_jobs=40, seed=123)

    first = generate_feasible_instance(config)
    second = generate_feasible_instance(config)

    pd.testing.assert_frame_equal(first.jobs_df, second.jobs_df)
    pd.testing.assert_frame_equal(first.hidden_schedule_df, second.hidden_schedule_df)


def test_generator_can_sample_from_alibaba_calibration(tmp_path):
    calibration_dir = tmp_path / "calibration"
    calibration_dir.mkdir()
    pd.DataFrame(
        [
            {
                "name": "sample-unconstrained",
                "pod_phase": "Succeeded",
                "qos": "BE",
                "gpu_type_required": "",
                "gpu_count_required": 1,
                "cpu_required": 4.0,
                "memory_required_gb": 16.0,
                "runtime_hours": 0.25,
                "duration": 1,
                "slot_minutes": 30,
                "wait_hours": 0.0,
                "creation_time": 0,
                "scheduled_time": 0,
                "deletion_time": 900,
            },
            {
                "name": "sample-t4",
                "pod_phase": "Succeeded",
                "qos": "BE",
                "gpu_type_required": "T4",
                "gpu_count_required": 2,
                "cpu_required": 8.0,
                "memory_required_gb": 32.0,
                "runtime_hours": 1.5,
                "duration": 2,
                "slot_minutes": 30,
                "wait_hours": 0.0,
                "creation_time": 0,
                "scheduled_time": 0,
                "deletion_time": 5400,
            },
        ]
    ).to_csv(calibration_dir / "alibaba_generator_job_samples.csv", index=False)

    instance = generate_feasible_instance(
        SyntheticInstanceConfig(
            num_jobs=20,
            seed=5,
            sampling_mode="alibaba",
            alibaba_calibration_dir=calibration_dir,
            slot_minutes=30,
            min_window_slack=1,
            max_window_slack=3,
        )
    )

    validate_jobs(instance.jobs_df)
    assert instance.config.delta_t == 0.5
    assert len(instance.hourly_df) == 48
    assert instance.hourly_df.loc[34, "hour_of_day"] == 17.0
    assert instance.hourly_df.loc[34, "grid_price"] == 145.0
    assert set(instance.jobs_df["gpu_type_required"]).issubset({"", "T4"})
    assert set(instance.jobs_df["source_trace_job"]).issubset({"sample-unconstrained", "sample-t4"})


def test_generator_rejects_alibaba_calibration_slot_mismatch(tmp_path):
    calibration_dir = tmp_path / "calibration"
    calibration_dir.mkdir()
    pd.DataFrame(
        [
            {
                "name": "sample",
                "gpu_type_required": "",
                "gpu_count_required": 1,
                "cpu_required": 4.0,
                "memory_required_gb": 16.0,
                "runtime_hours": 1.0,
                "duration": 1,
                "slot_minutes": 60,
            }
        ]
    ).to_csv(calibration_dir / "alibaba_generator_job_samples.csv", index=False)

    try:
        generate_feasible_instance(
            SyntheticInstanceConfig(
                num_jobs=1,
                sampling_mode="alibaba",
                alibaba_calibration_dir=calibration_dir,
                slot_minutes=30,
            )
        )
    except ValueError as exc:
        assert "slot_minutes" in str(exc)
    else:
        raise AssertionError("expected slot_minutes mismatch to fail")
