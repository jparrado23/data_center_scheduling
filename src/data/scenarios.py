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
            "compatible_categories": "fine_tuning,training,preprocessing",
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
            "compatible_categories": "fine_tuning,training,preprocessing",
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
            "compatible_categories": "fine_tuning,preprocessing",
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
                    "compatible_categories": "inference",
                },
        )
    clusters_df = pd.DataFrame(rows)
    validate_clusters(clusters_df)
    return clusters_df


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
    clusters_df = build_document_clusters()
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
    clusters_df = build_document_clusters()
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
