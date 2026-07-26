"""Build thesis-style scheduling scenarios from source data files."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import ModelConfig
from src.data.energy_prices import build_hourly_inputs_from_marginalpdbc
from src.data.job_instances import load_job_instance_csv
from src.data.solar_profiles import update_hourly_inputs_daily_solar, update_hourly_inputs_solar
from src.data.validation import validate_clusters, validate_hourly_inputs, validate_jobs


DEFAULT_FIXED_INFERENCE_BASELINE_MW = 0.010
CLUSTER_MODE_DOCUMENT = "document"
CLUSTER_MODE_ALIBABA_GPU_TYPES = "alibaba_gpu_types"
CLUSTER_MODES = {CLUSTER_MODE_DOCUMENT, CLUSTER_MODE_ALIBABA_GPU_TYPES}

ALIBABA_GPU_TYPE_COUNTS = {
    "A10": 2,
    "G2": 4392,
    "G3": 312,
    "P100": 265,
    "T4": 842,
    "V100M16": 195,
    "V100M32": 204,
}

DEFAULT_GPU_POWER_KW = {
    "A10": 0.150,
    "G2": 0.350,
    "G3": 0.350,
    "P100": 0.250,
    "T4": 0.070,
    "V100M16": 0.250,
    "V100M32": 0.250,
}

WORKLOAD_FILES = {
    "light": Path("data/instances/jobs_light.csv"),
    "tense": Path("data/instances/jobs_tense.csv"),
    "limit": Path("data/instances/jobs_limit.csv"),
}

ENERGY_SCENARIOS = {
    "clear_sky": {
        "date": "2023-04-04",
        "month": 4,
        "price_file": Path("docs/energy_price/marginalpdbc_20230404.1"),
    },
    "overcast": {
        "date": "2023-06-07",
        "month": 6,
        "price_file": Path("docs/energy_price/marginalpdbc_20230607.1"),
    },
    "base": {
        "date": "2023-09-27",
        "month": 9,
        "price_file": Path("docs/energy_price/marginalpdbc_20230927.1"),
    },
}

DEFAULT_MONTHLY_SOLAR_PROFILE = Path("data/solar_profile/monthly_solar_profiles.csv")


def _repo_path(project_root: str | Path, path: str | Path) -> Path:
    """Resolve a repo-relative path while preserving absolute paths."""

    resolved = Path(path)
    if resolved.is_absolute():
        return resolved
    return Path(project_root) / resolved


def build_document_clusters(*, include_inference_zone: bool = False) -> pd.DataFrame:
    """Return the cluster configuration described in the problem document.

    Zone A is excluded by default because inference is treated as fixed baseline
    load outside the flexible optimization.
    """

    rows = [
        {
            "cluster_id": "cluster_b",
            "cluster_role": "low_or_inference_gpu_pool",
            "capacity_kw": 54.6,
            "capacity": 0.0546,
            "power_capacity_kw": 54.6,
            "gpu_type": "T4",
            "gpu_count": 120,
            "gpu_capacity": 120,
            "cpu_capacity": 960,
            "memory_capacity_gb": 3840,
        },
        {
            "cluster_id": "cluster_c",
            "cluster_role": "mainstream_gpu_pool",
            "capacity_kw": 145.6,
            "capacity": 0.1456,
            "power_capacity_kw": 145.6,
            "gpu_type": "G2",
            "gpu_count": 160,
            "gpu_capacity": 160,
            "cpu_capacity": 1280,
            "memory_capacity_gb": 7680,
        },
        {
            "cluster_id": "cluster_d",
            "cluster_role": "high_perf_gpu_pool",
            "capacity_kw": 11.6,
            "capacity": 0.0116,
            "power_capacity_kw": 11.6,
            "gpu_type": "V100M32",
            "gpu_count": 32,
            "gpu_capacity": 32,
            "cpu_capacity": 256,
            "memory_capacity_gb": 1536,
        },
    ]
    if include_inference_zone:
        rows.insert(
            0,
                {
                    "cluster_id": "cluster_a",
                    "cluster_role": "serving_pool",
                    "capacity_kw": 10.0,
                    "capacity": 0.0100,
                    "power_capacity_kw": 10.0,
                    "gpu_type": "T4",
                    "gpu_count": 16,
                    "gpu_capacity": 16,
                    "cpu_capacity": 128,
                    "memory_capacity_gb": 512,
                    "reserved_for_online_inference": True,
                },
        )
    clusters_df = pd.DataFrame(rows)
    validate_clusters(clusters_df)
    return clusters_df


def build_alibaba_gpu_type_clusters(
    *,
    gpu_counts: dict[str, int] | None = None,
    gpu_power_kw: dict[str, float] | None = None,
    cpu_capacity: dict[str, float] | None = None,
    memory_capacity_gb: dict[str, float] | None = None,
    cpu_per_gpu: float = 8.0,
    memory_gb_per_gpu: float = 48.0,
    power_overhead_fraction: float = 0.0,
) -> pd.DataFrame:
    """Build one aggregate cluster per Alibaba GPU type.

    GPU counts use the Alibaba trace summary. CPU and memory can be supplied as
    per-GPU-type aggregate capacities from Alibaba node inventory; otherwise the
    fallback per-GPU assumptions are used. Power is IT-side capacity in MW; PUE
    is applied separately in the optimization.
    """

    counts = ALIBABA_GPU_TYPE_COUNTS if gpu_counts is None else gpu_counts
    powers = {**DEFAULT_GPU_POWER_KW, **(gpu_power_kw or {})}
    if cpu_per_gpu <= 0:
        raise ValueError("cpu_per_gpu must be positive")
    if memory_gb_per_gpu <= 0:
        raise ValueError("memory_gb_per_gpu must be positive")
    if power_overhead_fraction < 0:
        raise ValueError("power_overhead_fraction must be non-negative")

    rows = []
    for gpu_type, gpu_count in counts.items():
        if gpu_count <= 0:
            raise ValueError("GPU counts must be positive")
        if gpu_type not in powers:
            raise ValueError(f"missing per-GPU power assumption for {gpu_type}")
        power_capacity_kw = float(gpu_count) * float(powers[gpu_type]) * (1.0 + power_overhead_fraction)
        cluster_cpu_capacity = (
            float(cpu_capacity[gpu_type])
            if cpu_capacity is not None and gpu_type in cpu_capacity
            else float(gpu_count) * float(cpu_per_gpu)
        )
        cluster_memory_capacity_gb = (
            float(memory_capacity_gb[gpu_type])
            if memory_capacity_gb is not None and gpu_type in memory_capacity_gb
            else float(gpu_count) * float(memory_gb_per_gpu)
        )
        rows.append(
            {
                "cluster_id": f"cluster_{gpu_type.lower()}",
                "cluster_role": "alibaba_gpu_type_pool",
                "capacity_kw": power_capacity_kw,
                "capacity": power_capacity_kw / 1000.0,
                "power_capacity_kw": power_capacity_kw,
                "gpu_type": gpu_type,
                "gpu_count": int(gpu_count),
                "gpu_capacity": int(gpu_count),
                "cpu_capacity": cluster_cpu_capacity,
                "memory_capacity_gb": cluster_memory_capacity_gb,
            }
        )
    clusters_df = pd.DataFrame(rows)
    validate_clusters(clusters_df)
    return clusters_df


def build_clusters(cluster_mode: str = CLUSTER_MODE_ALIBABA_GPU_TYPES) -> pd.DataFrame:
    """Return the cluster table for a named modeling mode."""

    if cluster_mode == CLUSTER_MODE_DOCUMENT:
        return build_document_clusters()
    if cluster_mode == CLUSTER_MODE_ALIBABA_GPU_TYPES:
        return build_alibaba_gpu_type_clusters()
    raise ValueError(f"unknown cluster_mode {cluster_mode!r}; expected one of {sorted(CLUSTER_MODES)}")


def build_thesis_scenario(
    *,
    jobs_csv: str | Path,
    price_file: str | Path,
    solar_csv: str | Path,
    scenario_date: str,
    baseline_load_mw: float = DEFAULT_FIXED_INFERENCE_BASELINE_MW,
    renewable_price: float = 40.0,
    peak_price: float = 1000.0,
    contracted_power: float = 0.222,
    price_column: str = "last",
    solar_capacity_kwp: float | None = None,
    pue: float = 1.0,
    use_battery: bool = False,
    battery_power_capacity: float = 0.0,
    battery_energy_capacity: float = 0.0,
    battery_initial_soc: float = 0.0,
    battery_final_soc: float | None = None,
    battery_charge_efficiency: float = 0.95,
    battery_discharge_efficiency: float = 0.95,
    cluster_mode: str = CLUSTER_MODE_ALIBABA_GPU_TYPES,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, ModelConfig]:
    """Build one document-aligned scenario ready for the MILP solver."""

    jobs_df = load_job_instance_csv(jobs_csv)
    hourly_df = build_hourly_inputs_from_marginalpdbc(price_file, price_column=price_column)
    hourly_df = update_hourly_inputs_daily_solar(
        hourly_df,
        solar_csv,
        scenario_date,
        target_capacity_kwp=solar_capacity_kwp,
    )
    hourly_df["baseline_load"] = float(baseline_load_mw)
    clusters_df = build_clusters(cluster_mode)
    config = ModelConfig(
        contracted_power=float(contracted_power),
        renewable_price=float(renewable_price),
        peak_price=float(peak_price),
        delta_t=1.0,
        pue=float(pue),
        battery_power_capacity=float(battery_power_capacity) if use_battery else 0.0,
        battery_energy_capacity=float(battery_energy_capacity) if use_battery else 0.0,
        battery_initial_soc=float(battery_initial_soc) if use_battery else 0.0,
        battery_final_soc=float(battery_final_soc) if use_battery and battery_final_soc is not None else None,
        battery_charge_efficiency=float(battery_charge_efficiency),
        battery_discharge_efficiency=float(battery_discharge_efficiency),
    )

    validate_jobs(jobs_df)
    validate_hourly_inputs(hourly_df)
    validate_clusters(clusters_df)
    return jobs_df, hourly_df, clusters_df, config


def build_repo_scenario(
    *,
    project_root: str | Path,
    workload_case: str = "tense",
    jobs_csv: str | Path | None = None,
    energy_scenario: str = "base",
    solar_profile_csv: str | Path = DEFAULT_MONTHLY_SOLAR_PROFILE,
    baseline_load_mw: float = DEFAULT_FIXED_INFERENCE_BASELINE_MW,
    renewable_price: float = 40.0,
    peak_price: float = 1000.0,
    contracted_power: float = 0.222,
    price_column: str = "last",
    pue: float = 1.0,
    use_battery: bool = False,
    battery_power_capacity: float = 0.0,
    battery_energy_capacity: float = 0.0,
    battery_initial_soc: float = 0.0,
    battery_final_soc: float | None = None,
    battery_charge_efficiency: float = 0.95,
    battery_discharge_efficiency: float = 0.95,
    cluster_mode: str = CLUSTER_MODE_ALIBABA_GPU_TYPES,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, ModelConfig]:
    """Build one scenario from repo-local data files.

    This is the default path for notebooks and terminal solves. It combines a
    job-instance CSV, one OMIE price file, and the monthly average hourly solar
    profile for the price scenario's month. Pass `jobs_csv` to use a specific
    instance file instead of one of the named workload cases.
    """

    root = Path(project_root)
    if jobs_csv is None and workload_case not in WORKLOAD_FILES:
        raise ValueError(f"unknown workload_case {workload_case!r}; expected one of {sorted(WORKLOAD_FILES)}")
    if energy_scenario not in ENERGY_SCENARIOS:
        raise ValueError(f"unknown energy_scenario {energy_scenario!r}; expected one of {sorted(ENERGY_SCENARIOS)}")

    scenario = ENERGY_SCENARIOS[energy_scenario]
    jobs_path = _repo_path(root, jobs_csv) if jobs_csv is not None else _repo_path(root, WORKLOAD_FILES[workload_case])
    jobs_df = load_job_instance_csv(jobs_path)
    hourly_df = build_hourly_inputs_from_marginalpdbc(
        _repo_path(root, scenario["price_file"]),
        price_column=price_column,
    )
    solar_profile_df = pd.read_csv(_repo_path(root, solar_profile_csv))
    hourly_df = update_hourly_inputs_solar(hourly_df, solar_profile_df, int(scenario["month"]))
    hourly_df["baseline_load"] = float(baseline_load_mw)
    clusters_df = build_clusters(cluster_mode)
    config = ModelConfig(
        contracted_power=float(contracted_power),
        renewable_price=float(renewable_price),
        peak_price=float(peak_price),
        delta_t=1.0,
        pue=float(pue),
        battery_power_capacity=float(battery_power_capacity) if use_battery else 0.0,
        battery_energy_capacity=float(battery_energy_capacity) if use_battery else 0.0,
        battery_initial_soc=float(battery_initial_soc) if use_battery else 0.0,
        battery_final_soc=float(battery_final_soc) if use_battery and battery_final_soc is not None else None,
        battery_charge_efficiency=float(battery_charge_efficiency),
        battery_discharge_efficiency=float(battery_discharge_efficiency),
    )

    validate_jobs(jobs_df)
    validate_hourly_inputs(hourly_df)
    validate_clusters(clusters_df)
    return jobs_df, hourly_df, clusters_df, config
