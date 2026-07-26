import pandas as pd

from src.instance_generator.alibaba_calibration import (
    AlibabaCalibrationConfig,
    build_alibaba_generator_calibration,
    normalize_gpu_spec,
    prepare_alibaba_generator_samples,
    summarize_gpu_type_cluster_capacity,
)


def _pods() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "name": "j1",
                "cpu_milli": 4000,
                "memory_mib": 8192,
                "num_gpu": 1,
                "gpu_milli": 1000,
                "gpu_spec": "",
                "qos": "BE",
                "pod_phase": "Succeeded",
                "creation_time": 0,
                "scheduled_time": 0,
                "deletion_time": 1800,
            },
            {
                "name": "j2",
                "cpu_milli": 8000,
                "memory_mib": 16384,
                "num_gpu": 2,
                "gpu_milli": 2000,
                "gpu_spec": "V100M32|T4|T4",
                "qos": "BE",
                "pod_phase": "Succeeded",
                "creation_time": 3600,
                "scheduled_time": 3600,
                "deletion_time": 10800,
            },
            {
                "name": "too-long",
                "cpu_milli": 8000,
                "memory_mib": 16384,
                "num_gpu": 1,
                "gpu_milli": 1000,
                "gpu_spec": "G2",
                "qos": "BE",
                "pod_phase": "Running",
                "creation_time": 0,
                "scheduled_time": 0,
                "deletion_time": 10 * 3600,
            },
            {
                "name": "no-gpu",
                "cpu_milli": 1000,
                "memory_mib": 1024,
                "num_gpu": 0,
                "gpu_milli": 0,
                "gpu_spec": "",
                "qos": "BE",
                "pod_phase": "Succeeded",
                "creation_time": 0,
                "scheduled_time": 0,
                "deletion_time": 3600,
            },
        ]
    )


def _gpu_nodes() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "sn": "node-g2-1",
                "cpu_milli": 96000,
                "memory_mib": 393216,
                "gpu": 8,
                "model": "G2",
            },
            {
                "sn": "node-g2-2",
                "cpu_milli": 96000,
                "memory_mib": 393216,
                "gpu": 8,
                "model": "G2",
            },
            {
                "sn": "node-t4-1",
                "cpu_milli": 104000,
                "memory_mib": 524288,
                "gpu": 2,
                "model": "T4",
            },
        ]
    )


def test_prepare_alibaba_generator_samples_filters_and_converts_slots():
    samples = prepare_alibaba_generator_samples(_pods(), max_runtime_hours=4.0, slot_minutes=30)

    assert samples["name"].tolist() == ["j1", "j2"]
    assert samples["duration"].tolist() == [1, 4]
    assert samples.loc[0, "gpu_type_required"] == ""
    assert samples.loc[1, "gpu_type_required"] == "T4|V100M32"
    assert samples.loc[1, "memory_required_gb"] == 16.0


def test_normalize_gpu_spec_deduplicates_and_sorts_tokens():
    assert normalize_gpu_spec("V100M32|T4|T4") == "T4|V100M32"
    assert normalize_gpu_spec("") == ""


def test_summarize_gpu_type_cluster_capacity_aggregates_node_inventory():
    summary = summarize_gpu_type_cluster_capacity(_gpu_nodes()).set_index("gpu_type")

    assert summary.loc["G2", "nodes"] == 2
    assert summary.loc["G2", "gpu_count"] == 16
    assert summary.loc["G2", "cpu_capacity"] == 192.0
    assert summary.loc["G2", "memory_capacity_gb"] == 768.0
    assert summary.loc["G2", "cpu_per_gpu"] == 12.0
    assert summary.loc["T4", "memory_gb_per_gpu"] == 256.0


def test_build_alibaba_generator_calibration_writes_expected_tables(tmp_path):
    raw_dir = tmp_path / "raw"
    output_dir = tmp_path / "outputs"
    raw_dir.mkdir()
    _pods().to_csv(raw_dir / "openb_pod_list_gpuspec33.csv", index=False)
    _gpu_nodes().to_csv(raw_dir / "openb_node_list_gpu_node.csv", index=False)

    outputs = build_alibaba_generator_calibration(
        AlibabaCalibrationConfig(raw_dir=raw_dir, output_dir=output_dir, max_runtime_hours=4.0, slot_minutes=30)
    )

    assert outputs["alibaba_generator_job_samples"].shape[0] == 2
    assert (output_dir / "alibaba_generator_job_samples.csv").exists()
    assert (output_dir / "alibaba_jobs_per_24h.csv").exists()
    assert (output_dir / "alibaba_cluster_capacity_from_nodes.csv").exists()
    assert outputs["alibaba_cluster_capacity_from_nodes"].set_index("gpu_type").loc["G2", "gpu_count"] == 16
    summary = outputs["alibaba_generator_calibration_summary"].set_index("metric")["value"]
    assert summary["generator_sample_jobs"] == 2
    assert "filtered_jobs_per_24h_mean_by_day" in summary
