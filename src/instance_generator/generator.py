"""Synthetic feasible-by-construction instance generator.

The generator first creates a hidden schedule that obeys cluster power and GPU
limits. It then expands each hidden start time into a flexible time window and
exports only the job data required by the MILP. The hidden schedule is returned
for verification and experiment diagnostics, but it is not used by the solver.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import ModelConfig
from src.data.scenarios import build_alibaba_gpu_type_clusters
from src.data.validation import validate_clusters, validate_hourly_inputs, validate_jobs


DEFAULT_ALIBABA_CALIBRATION_DIR = Path("experiments/outputs/alibaba_2023_eda")
ALIBABA_JOB_SAMPLES_FILE = "alibaba_generator_job_samples.csv"
SAMPLING_MODE_ALIBABA = "alibaba"
SAMPLING_MODE_PARAMETRIC = "parametric"
SAMPLING_MODES = {SAMPLING_MODE_ALIBABA, SAMPLING_MODE_PARAMETRIC}


ALPHA_COLUMNS_BY_CLUSTER = {
    "cluster_b": "alpha_B",
    "cluster_c": "alpha_C",
    "cluster_d": "alpha_D",
}


@dataclass(frozen=True)
class SyntheticInstanceConfig:
    """Controls one feasible synthetic scheduling instance."""

    num_jobs: int
    horizon_hours: int = 24
    slot_minutes: int = 60
    seed: int = 42
    sampling_mode: str = SAMPLING_MODE_ALIBABA
    alibaba_calibration_dir: Path = DEFAULT_ALIBABA_CALIBRATION_DIR
    min_duration: int = 1
    max_duration: int = 6
    min_gpu_demand: int = 1
    max_gpu_demand: int = 4
    power_per_gpu_kw: float = 0.35
    cpu_per_gpu: float = 8.0
    memory_gb_per_gpu: float = 48.0
    power_jitter_fraction: float = 0.20
    min_window_slack: int = 2
    max_window_slack: int = 8
    extra_compatibility_probability: float = 0.75
    max_placement_attempts_per_job: int = 2_000
    renewable_price: float = 40.0
    peak_price: float = 1000.0
    contracted_power: float = 0.222
    baseline_load_mw: float = 0.010
    pue: float = 1.2

    def __post_init__(self) -> None:
        if self.num_jobs <= 0:
            raise ValueError("num_jobs must be positive")
        if self.horizon_hours <= 0:
            raise ValueError("horizon_hours must be positive")
        if self.slot_minutes <= 0:
            raise ValueError("slot_minutes must be positive")
        if 60 % self.slot_minutes != 0:
            raise ValueError("slot_minutes must divide one hour")
        if self.sampling_mode not in SAMPLING_MODES:
            raise ValueError(f"sampling_mode must be one of {sorted(SAMPLING_MODES)}")
        if self.min_duration <= 0 or self.max_duration < self.min_duration:
            raise ValueError("duration bounds must be positive and ordered")
        if self.max_duration > self.horizon_slots:
            raise ValueError("max_duration cannot exceed the horizon length in slots")
        if self.min_gpu_demand <= 0 or self.max_gpu_demand < self.min_gpu_demand:
            raise ValueError("GPU-demand bounds must be positive and ordered")
        if self.power_per_gpu_kw <= 0:
            raise ValueError("power_per_gpu_kw must be positive")
        if self.cpu_per_gpu <= 0:
            raise ValueError("cpu_per_gpu must be positive")
        if self.memory_gb_per_gpu <= 0:
            raise ValueError("memory_gb_per_gpu must be positive")
        if not 0 <= self.power_jitter_fraction <= 1:
            raise ValueError("power_jitter_fraction must be between 0 and 1")
        if self.min_window_slack < 0 or self.max_window_slack < self.min_window_slack:
            raise ValueError("window-slack bounds must be non-negative and ordered")
        if not 0 <= self.extra_compatibility_probability <= 1:
            raise ValueError("extra_compatibility_probability must be between 0 and 1")
        if self.max_placement_attempts_per_job <= 0:
            raise ValueError("max_placement_attempts_per_job must be positive")

    @property
    def horizon_slots(self) -> int:
        return int(self.horizon_hours * 60 / self.slot_minutes)


@dataclass(frozen=True)
class FeasibilityReport:
    """Diagnostics proving the generated instance has at least one schedule."""

    total_job_gpu_hours: float
    available_gpu_hours: float
    gpu_utilization_ratio: float
    total_job_mwh: float
    available_cluster_mwh: float
    power_utilization_ratio: float
    max_cluster_gpu_utilization: float
    max_cluster_power_utilization: float


@dataclass(frozen=True)
class SyntheticInstance:
    """Generated dataframes and metadata for one stress-test instance."""

    jobs_df: pd.DataFrame
    hourly_df: pd.DataFrame
    clusters_df: pd.DataFrame
    config: ModelConfig
    hidden_schedule_df: pd.DataFrame
    feasibility_report: FeasibilityReport


def _cluster_lookup(clusters_df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    clusters: dict[str, dict[str, Any]] = {}
    for row in clusters_df.itertuples(index=False):
        gpu_capacity = int(row.gpu_count) if hasattr(row, "gpu_count") else int(row.gpu_capacity)
        clusters[str(row.cluster_id)] = {
            "capacity": float(row.capacity),
            "gpu_capacity": gpu_capacity,
            "gpu_type": str(getattr(row, "gpu_type", "")).strip(),
            "cpu_capacity": float(getattr(row, "cpu_capacity", 0.0)),
            "memory_capacity_gb": float(getattr(row, "memory_capacity_gb", 0.0)),
        }
    return clusters


def _load_alibaba_job_samples(calibration_dir: str | Path) -> pd.DataFrame:
    path = Path(calibration_dir) / ALIBABA_JOB_SAMPLES_FILE
    if not path.exists():
        raise FileNotFoundError(
            f"Alibaba calibration samples not found: {path}. "
            "Run notebooks/06_alibaba_2023_trace_eda.ipynb or build_alibaba_generator_calibration first."
        )
    samples = pd.read_csv(path)
    required_columns = {
        "gpu_type_required",
        "gpu_count_required",
        "cpu_required",
        "memory_required_gb",
        "duration",
    }
    missing = required_columns.difference(samples.columns)
    if missing:
        raise ValueError(f"Alibaba calibration samples are missing columns: {sorted(missing)}")
    if samples.empty:
        raise ValueError("Alibaba calibration samples must contain at least one row")
    return samples


def _validate_calibration_slot_minutes(samples: pd.DataFrame, slot_minutes: int) -> None:
    if "slot_minutes" not in samples.columns:
        raise ValueError(
            "Alibaba calibration samples do not include slot_minutes metadata; "
            "regenerate calibration outputs before using non-parametric sampling"
        )
    calibration_slots = set(samples["slot_minutes"].dropna().astype(int).tolist())
    if calibration_slots != {int(slot_minutes)}:
        raise ValueError(
            f"Alibaba calibration slot_minutes {sorted(calibration_slots)} do not match "
            f"generator slot_minutes {slot_minutes}; regenerate calibration with the same slot size"
        )


def _allowed_clusters_for_gpu_spec(
    gpu_type_required: str,
    clusters: list[str],
    cluster_data: dict[str, dict[str, Any]],
) -> list[str]:
    if not gpu_type_required:
        return clusters
    allowed_types = {token.strip() for token in str(gpu_type_required).split("|") if token.strip()}
    return [cluster for cluster in clusters if cluster_data[cluster]["gpu_type"] in allowed_types]


def _cluster_power_per_gpu_kw(cluster: str, cluster_data: dict[str, dict[str, Any]]) -> float:
    gpu_capacity = max(cluster_data[cluster]["gpu_capacity"], 1)
    return cluster_data[cluster]["capacity"] * 1000.0 / gpu_capacity


def _generate_slot_aware_hourly_inputs(horizon_slots: int, slot_minutes: int) -> pd.DataFrame:
    slots = np.arange(horizon_slots)
    time_hours = slots * slot_minutes / 60.0
    hour_of_day = time_hours % 24.0
    renewable_available = np.maximum(0.0, 0.08 * np.sin((hour_of_day - 6.0) / 12.0 * np.pi))
    grid_price = np.where((hour_of_day >= 17.0) & (hour_of_day < 22.0), 145.0, 85.0)
    grid_price = np.where((hour_of_day >= 0.0) & (hour_of_day < 6.0), 65.0, grid_price)
    return pd.DataFrame(
        {
            "hour": slots.astype(int),
            "time_hours": time_hours,
            "hour_of_day": hour_of_day,
            "renewable_available": renewable_available.round(3),
            "grid_price": grid_price.astype(float),
        }
    )


def _sample_category(compatible_count: int) -> str:
    if compatible_count >= 3:
        return "preprocessing"
    if compatible_count == 2:
        return "training"
    return "fine_tuning"


def _build_compatible_clusters(
    rng: np.random.Generator,
    clusters: list[str],
    hidden_cluster: str,
    extra_probability: float,
) -> list[str]:
    compatible = {hidden_cluster}
    for cluster in clusters:
        if cluster != hidden_cluster and rng.random() < extra_probability:
            compatible.add(cluster)
    return sorted(compatible)


def _sample_job_profile(
    rng: np.random.Generator,
    instance_config: SyntheticInstanceConfig,
    alibaba_samples: pd.DataFrame | None,
) -> dict[str, Any]:
    if instance_config.sampling_mode == SAMPLING_MODE_ALIBABA:
        if alibaba_samples is None:
            raise ValueError("alibaba_samples must be provided when sampling_mode='alibaba'")
        sample = alibaba_samples.iloc[int(rng.integers(0, len(alibaba_samples)))]
        duration = int(sample.duration)
        duration = max(1, min(duration, instance_config.horizon_slots))
        return {
            "duration": duration,
            "gpus": max(1, int(sample.gpu_count_required)),
            "gpu_type_required": "" if pd.isna(sample.gpu_type_required) else str(sample.gpu_type_required),
            "cpu": max(0.0, float(sample.cpu_required)),
            "memory_gb": max(0.0, float(sample.memory_required_gb)),
            "runtime_hours_sampled": float(
                sample["runtime_hours"] if "runtime_hours" in sample else duration * instance_config.slot_minutes / 60.0
            ),
            "source_trace_job": str(sample["name"] if "name" in sample else ""),
        }

    duration = int(rng.integers(instance_config.min_duration, instance_config.max_duration + 1))
    gpus = int(rng.integers(instance_config.min_gpu_demand, instance_config.max_gpu_demand + 1))
    return {
        "duration": duration,
        "gpus": gpus,
        "gpu_type_required": None,
        "cpu": float(gpus * instance_config.cpu_per_gpu),
        "memory_gb": float(gpus * instance_config.memory_gb_per_gpu),
        "runtime_hours_sampled": duration * instance_config.slot_minutes / 60.0,
        "source_trace_job": "",
    }


def _can_place(
    *,
    cluster: str,
    start: int,
    duration: int,
    power_mw: float,
    gpus: int,
    cpu: float,
    memory_gb: float,
    power_used: dict[str, np.ndarray],
    gpu_used: dict[str, np.ndarray],
    cpu_used: dict[str, np.ndarray],
    memory_used: dict[str, np.ndarray],
    cluster_data: dict[str, dict[str, Any]],
) -> bool:
    stop = start + duration
    if np.any(power_used[cluster][start:stop] + power_mw > cluster_data[cluster]["capacity"] + 1e-12):
        return False
    if np.any(gpu_used[cluster][start:stop] + gpus > cluster_data[cluster]["gpu_capacity"]):
        return False
    if cluster_data[cluster]["cpu_capacity"] > 0 and np.any(
        cpu_used[cluster][start:stop] + cpu > cluster_data[cluster]["cpu_capacity"]
    ):
        return False
    if cluster_data[cluster]["memory_capacity_gb"] > 0 and np.any(
        memory_used[cluster][start:stop] + memory_gb > cluster_data[cluster]["memory_capacity_gb"]
    ):
        return False
    return True


def _place_job(
    *,
    rng: np.random.Generator,
    duration: int,
    power_mw: float,
    gpus: int,
    cpu: float,
    memory_gb: float,
    clusters: list[str],
    horizon_hours: int,
    power_used: dict[str, np.ndarray],
    gpu_used: dict[str, np.ndarray],
    cpu_used: dict[str, np.ndarray],
    memory_used: dict[str, np.ndarray],
    cluster_data: dict[str, dict[str, Any]],
    max_attempts: int,
) -> tuple[str, int]:
    for _ in range(max_attempts):
        cluster = str(rng.choice(clusters))
        start = int(rng.integers(0, horizon_hours - duration + 1))
        if _can_place(
            cluster=cluster,
            start=start,
            duration=duration,
            power_mw=power_mw,
            gpus=gpus,
            cpu=cpu,
            memory_gb=memory_gb,
            power_used=power_used,
            gpu_used=gpu_used,
            cpu_used=cpu_used,
            memory_used=memory_used,
            cluster_data=cluster_data,
        ):
            return cluster, start

    ordered_clusters = list(clusters)
    rng.shuffle(ordered_clusters)
    for cluster in ordered_clusters:
        starts = list(range(horizon_hours - duration + 1))
        rng.shuffle(starts)
        for start in starts:
            if _can_place(
                cluster=cluster,
                start=start,
                duration=duration,
                power_mw=power_mw,
                gpus=gpus,
                cpu=cpu,
                memory_gb=memory_gb,
                power_used=power_used,
                gpu_used=gpu_used,
                cpu_used=cpu_used,
                memory_used=memory_used,
                cluster_data=cluster_data,
            ):
                return cluster, start

    raise ValueError(
        "could not place a generated job without violating cluster power/GPU/CPU/memory capacity; "
        "reduce num_jobs, duration, GPU demand, or power_per_gpu_kw"
    )


def _build_feasibility_report(
    jobs_df: pd.DataFrame,
    clusters_df: pd.DataFrame,
    power_used: dict[str, np.ndarray],
    gpu_used: dict[str, np.ndarray],
    delta_t: float,
) -> FeasibilityReport:
    total_job_gpu_hours = float((jobs_df["gpus"] * jobs_df["duration"] * delta_t).sum())
    total_job_mwh = float((jobs_df["power"] * jobs_df["duration"] * delta_t).sum())
    gpu_capacity_column = "gpu_count" if "gpu_count" in clusters_df.columns else "gpu_capacity"
    available_gpu_hours = float((clusters_df[gpu_capacity_column] * len(next(iter(gpu_used.values()))) * delta_t).sum())
    available_cluster_mwh = float((clusters_df["capacity"] * len(next(iter(power_used.values()))) * delta_t).sum())

    max_gpu_ratio = max(
        float(gpu_used[str(row.cluster_id)].max() / getattr(row, gpu_capacity_column))
        for row in clusters_df.itertuples(index=False)
    )
    max_power_ratio = max(
        float(power_used[str(row.cluster_id)].max() / row.capacity)
        for row in clusters_df.itertuples(index=False)
    )

    return FeasibilityReport(
        total_job_gpu_hours=total_job_gpu_hours,
        available_gpu_hours=available_gpu_hours,
        gpu_utilization_ratio=total_job_gpu_hours / available_gpu_hours,
        total_job_mwh=total_job_mwh,
        available_cluster_mwh=available_cluster_mwh,
        power_utilization_ratio=total_job_mwh / available_cluster_mwh,
        max_cluster_gpu_utilization=max_gpu_ratio,
        max_cluster_power_utilization=max_power_ratio,
    )


def generate_feasible_instance(
    instance_config: SyntheticInstanceConfig,
    *,
    clusters_df: pd.DataFrame | None = None,
) -> SyntheticInstance:
    """Generate a MILP-ready synthetic instance with a known feasible schedule."""

    rng = np.random.default_rng(instance_config.seed)
    clusters_df = build_alibaba_gpu_type_clusters() if clusters_df is None else clusters_df.copy()
    validate_clusters(clusters_df)
    if "gpu_capacity" not in clusters_df.columns and "gpu_count" not in clusters_df.columns:
        raise ValueError("clusters_df must include gpu_capacity or gpu_count for feasible instance generation")

    cluster_data = _cluster_lookup(clusters_df)
    clusters = sorted(cluster_data)
    horizon_slots = instance_config.horizon_slots
    alibaba_samples = (
        _load_alibaba_job_samples(instance_config.alibaba_calibration_dir)
        if instance_config.sampling_mode == SAMPLING_MODE_ALIBABA
        else None
    )
    if alibaba_samples is not None:
        _validate_calibration_slot_minutes(alibaba_samples, instance_config.slot_minutes)
    power_used = {cluster: np.zeros(horizon_slots) for cluster in clusters}
    gpu_used = {cluster: np.zeros(horizon_slots, dtype=int) for cluster in clusters}
    cpu_used = {cluster: np.zeros(horizon_slots) for cluster in clusters}
    memory_used = {cluster: np.zeros(horizon_slots) for cluster in clusters}

    job_rows: list[dict[str, Any]] = []
    hidden_rows: list[dict[str, Any]] = []

    for index in range(instance_config.num_jobs):
        profile = _sample_job_profile(rng, instance_config, alibaba_samples)
        duration = int(profile["duration"])
        gpus = int(profile["gpus"])
        cpu = float(profile["cpu"])
        memory_gb = float(profile["memory_gb"])
        requested_gpu_spec = profile["gpu_type_required"]
        candidate_clusters = (
            clusters
            if requested_gpu_spec is None
            else _allowed_clusters_for_gpu_spec(requested_gpu_spec, clusters, cluster_data)
        )
        candidate_clusters = [
            cluster
            for cluster in candidate_clusters
            if gpus <= cluster_data[cluster]["gpu_capacity"]
            and (cluster_data[cluster]["cpu_capacity"] <= 0 or cpu <= cluster_data[cluster]["cpu_capacity"])
            and (
                cluster_data[cluster]["memory_capacity_gb"] <= 0
                or memory_gb <= cluster_data[cluster]["memory_capacity_gb"]
            )
        ]
        if not candidate_clusters:
            raise ValueError(
                "sampled job has no feasible GPU-type pool before placement; "
                "check Alibaba calibration, GPU counts, CPU/memory assumptions, and cluster capacities"
            )
        hidden_cluster, hidden_start = _place_job(
            rng=rng,
            duration=duration,
            power_mw=0.0,
            gpus=gpus,
            cpu=cpu,
            memory_gb=memory_gb,
            clusters=candidate_clusters,
            horizon_hours=horizon_slots,
            power_used=power_used,
            gpu_used=gpu_used,
            cpu_used=cpu_used,
            memory_used=memory_used,
            cluster_data=cluster_data,
            max_attempts=instance_config.max_placement_attempts_per_job,
        )
        jitter_low = 1.0 - instance_config.power_jitter_fraction
        jitter_high = 1.0 + instance_config.power_jitter_fraction
        per_gpu_power_kw = (
            _cluster_power_per_gpu_kw(hidden_cluster, cluster_data)
            if instance_config.sampling_mode == SAMPLING_MODE_ALIBABA
            else instance_config.power_per_gpu_kw
        )
        power_kw = float(gpus * per_gpu_power_kw * rng.uniform(jitter_low, jitter_high))
        power_mw = power_kw / 1000.0
        if np.any(
            power_used[hidden_cluster][hidden_start : hidden_start + duration] + power_mw
            > cluster_data[hidden_cluster]["capacity"] + 1e-12
        ):
            hidden_cluster, hidden_start = _place_job(
                rng=rng,
                duration=duration,
                power_mw=power_mw,
                gpus=gpus,
                cpu=cpu,
                memory_gb=memory_gb,
                clusters=candidate_clusters,
                horizon_hours=horizon_slots,
                power_used=power_used,
                gpu_used=gpu_used,
                cpu_used=cpu_used,
                memory_used=memory_used,
                cluster_data=cluster_data,
                max_attempts=instance_config.max_placement_attempts_per_job,
            )

        power_used[hidden_cluster][hidden_start : hidden_start + duration] += power_mw
        gpu_used[hidden_cluster][hidden_start : hidden_start + duration] += gpus
        cpu_used[hidden_cluster][hidden_start : hidden_start + duration] += cpu
        memory_used[hidden_cluster][hidden_start : hidden_start + duration] += memory_gb

        if requested_gpu_spec is None:
            compatible_clusters = _build_compatible_clusters(
                rng,
                clusters,
                hidden_cluster,
                instance_config.extra_compatibility_probability,
            )
            allowed_gpu_types = sorted(
                {cluster_data[cluster]["gpu_type"] for cluster in compatible_clusters if cluster_data[cluster]["gpu_type"]}
            )
        else:
            compatible_clusters = candidate_clusters
            allowed_gpu_types = [] if requested_gpu_spec == "" else requested_gpu_spec.split("|")
        slack_before = int(rng.integers(instance_config.min_window_slack, instance_config.max_window_slack + 1))
        slack_after = int(rng.integers(instance_config.min_window_slack, instance_config.max_window_slack + 1))
        earliest_start = max(0, hidden_start - slack_before)
        latest_start = min(horizon_slots - duration, hidden_start + slack_after)

        job_id = f"synthetic_{index + 1:05d}"
        workload_family = _sample_category(len(compatible_clusters))
        row: dict[str, Any] = {
            "job_id": job_id,
            "category": workload_family,
            "workload_family": workload_family,
            "gpus": gpus,
            "gpu_count_required": gpus,
            "gpu_type_required": "|".join(allowed_gpu_types),
            "cpu_required": cpu,
            "memory_required_gb": memory_gb,
            "duration": duration,
            "duration_h": duration * instance_config.slot_minutes / 60.0,
            "power_kw": round(power_kw, 6),
            "e_kw": round(power_kw, 6),
            "power": round(power_mw, 9),
            "earliest_start": earliest_start,
            "latest_start": latest_start,
            "slot_minutes": instance_config.slot_minutes,
            "runtime_hours_sampled": round(float(profile["runtime_hours_sampled"]), 6),
            "source_trace_job": profile["source_trace_job"],
        }
        for cluster, alpha_column in ALPHA_COLUMNS_BY_CLUSTER.items():
            if cluster in cluster_data:
                row[alpha_column] = int(cluster in compatible_clusters)
        job_rows.append(row)
        hidden_rows.append(
            {
                "job_id": job_id,
                "assigned_cluster": hidden_cluster,
                "start_hour": hidden_start,
                "end_hour": hidden_start + duration,
                "start_slot": hidden_start,
                "end_slot": hidden_start + duration,
                "duration": duration,
                "duration_h": duration * instance_config.slot_minutes / 60.0,
                "power": round(power_mw, 9),
                "gpus": gpus,
                "gpu_count_required": gpus,
                "gpu_type_required": "|".join(allowed_gpu_types),
                "cpu_required": cpu,
                "memory_required_gb": memory_gb,
                "slot_minutes": instance_config.slot_minutes,
            }
        )

    jobs_df = pd.DataFrame(job_rows)
    hidden_schedule_df = pd.DataFrame(hidden_rows)
    hourly_df = _generate_slot_aware_hourly_inputs(horizon_slots, instance_config.slot_minutes)
    hourly_df["baseline_load"] = float(instance_config.baseline_load_mw)
    config = ModelConfig(
        contracted_power=instance_config.contracted_power,
        renewable_price=instance_config.renewable_price,
        peak_price=instance_config.peak_price,
        delta_t=instance_config.slot_minutes / 60.0,
        pue=instance_config.pue,
    )
    report = _build_feasibility_report(jobs_df, clusters_df, power_used, gpu_used, config.delta_t)

    validate_jobs(jobs_df)
    validate_hourly_inputs(hourly_df)
    validate_clusters(clusters_df)
    return SyntheticInstance(
        jobs_df=jobs_df,
        hourly_df=hourly_df,
        clusters_df=clusters_df,
        config=config,
        hidden_schedule_df=hidden_schedule_df,
        feasibility_report=report,
    )
