import json

import numpy as np
import pandas as pd

from src.data.validation import validate_clusters, validate_hourly_inputs, validate_jobs
from src.instance_generator.hard_instances import (
    BenchmarkInstanceSpec,
    DifficultyConfig,
    build_energy_inputs,
    build_small_clusters,
    create_hard_benchmark_pack,
    generate_difficulty_instance,
    instance_diagnostics,
)


def test_generate_difficulty_instance_returns_valid_model_inputs():
    config = DifficultyConfig(
        target_gpu_utilization=0.75,
        cheap_block_pressure=1.4,
        restricted_gpu_share=0.5,
        multi_gpu_share=0.4,
        max_jobs=70,
        max_generation_attempts=5_000,
    )
    clusters = build_small_clusters()
    jobs_df, hidden_df = generate_difficulty_instance(config, seed=123, clusters=clusters)
    hourly_df, model_config = build_energy_inputs(config, clusters)

    validate_jobs(jobs_df)
    validate_clusters(clusters)
    validate_hourly_inputs(hourly_df)

    assert len(jobs_df) == len(hidden_df)
    assert model_config.delta_t == 0.25
    assert (jobs_df["gpu_count_required"] >= 1).all()
    assert (jobs_df["latest_start"] + jobs_df["duration"] <= config.horizon_slots).all()
    assert (jobs_df["gpu_count_required"] > 1).any()
    assert (jobs_df["gpu_type_required"].astype(str).str.len() > 0).any()


def test_hidden_schedule_obeys_resource_capacity():
    config = DifficultyConfig(target_gpu_utilization=0.80, max_jobs=70, max_generation_attempts=5_000)
    clusters = build_small_clusters().set_index("cluster_id")
    jobs_df, hidden_df = generate_difficulty_instance(config, seed=456, clusters=clusters.reset_index())
    jobs = jobs_df.set_index("job_id")
    usage = {
        cluster_id: {
            "gpu": np.zeros(config.horizon_slots),
            "cpu": np.zeros(config.horizon_slots),
            "memory": np.zeros(config.horizon_slots),
            "power": np.zeros(config.horizon_slots),
        }
        for cluster_id in clusters.index
    }

    for row in hidden_df.itertuples(index=False):
        job = jobs.loc[row.job_id]
        assert job.earliest_start <= row.hidden_start <= job.latest_start
        end = row.hidden_start + row.duration
        usage[row.hidden_cluster]["gpu"][row.hidden_start:end] += job.gpu_count_required
        usage[row.hidden_cluster]["cpu"][row.hidden_start:end] += job.cpu_required
        usage[row.hidden_cluster]["memory"][row.hidden_start:end] += job.memory_required_gb
        usage[row.hidden_cluster]["power"][row.hidden_start:end] += job.power

    for cluster_id, cluster_usage in usage.items():
        assert cluster_usage["gpu"].max() <= clusters.loc[cluster_id, "gpu_count"]
        assert cluster_usage["cpu"].max() <= clusters.loc[cluster_id, "cpu_capacity"]
        assert cluster_usage["memory"].max() <= clusters.loc[cluster_id, "memory_capacity_gb"]
        assert cluster_usage["power"].max() <= clusters.loc[cluster_id, "capacity"]


def test_instance_diagnostics_report_actual_pressure_metrics():
    config = DifficultyConfig(target_gpu_utilization=0.75, cheap_block_pressure=1.4, max_jobs=70)
    clusters = build_small_clusters()
    jobs_df, _ = generate_difficulty_instance(config, seed=789, clusters=clusters)
    diagnostics = instance_diagnostics(jobs_df, config, clusters)

    assert diagnostics["num_jobs"] == len(jobs_df)
    assert diagnostics["global_gpu_slot_utilization"] > 0
    assert diagnostics["cheap_block_pressure_actual"] > 0
    assert "assignment_density_proxy" in diagnostics


def test_create_hard_benchmark_pack_writes_manifest_and_instance_files(tmp_path):
    config = DifficultyConfig(target_gpu_utilization=0.65, cheap_block_pressure=1.2, max_jobs=50)
    specs = [BenchmarkInstanceSpec("test_hard_family", config, (10, 11))]

    manifest = create_hard_benchmark_pack(tmp_path, specs=specs)

    assert len(manifest) == 2
    assert (tmp_path / "manifest.csv").exists()
    for row in manifest.itertuples(index=False):
        instance_dir = tmp_path / row.instance_name
        assert instance_dir.exists()
        for filename in ("jobs.csv", "hourly.csv", "clusters.csv", "hidden_feasible_schedule.csv", "metadata.json"):
            assert (instance_dir / filename).exists()
        metadata = json.loads((instance_dir / "metadata.json").read_text())
        assert metadata["label"] == "test_hard_family"
        validate_jobs(pd.read_csv(instance_dir / "jobs.csv"))
