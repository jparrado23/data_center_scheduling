"""Build Alibaba trace calibration tables for synthetic instance generation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_RAW_DIR = Path("data/raw/alibaba_gpu_v2023")
DEFAULT_OUTPUT_DIR = Path("experiments/outputs/alibaba_2023_eda")
POD_TRACE_FILE = "openb_pod_list_gpuspec33.csv"
SECONDS_PER_DAY = 24 * 60 * 60


@dataclass(frozen=True)
class AlibabaCalibrationConfig:
    """Parameters controlling trace filtering and slot conversion."""

    raw_dir: Path = DEFAULT_RAW_DIR
    output_dir: Path = DEFAULT_OUTPUT_DIR
    max_runtime_hours: float = 4.0
    slot_minutes: int = 60

    def __post_init__(self) -> None:
        if self.max_runtime_hours <= 0:
            raise ValueError("max_runtime_hours must be positive")
        if self.slot_minutes <= 0:
            raise ValueError("slot_minutes must be positive")
        if 60 % self.slot_minutes != 0:
            raise ValueError("slot_minutes must divide one hour")


def normalize_gpu_spec(value: object) -> str:
    """Normalize Alibaba GPU-spec strings for generator compatibility."""

    if pd.isna(value) or str(value).strip() == "":
        return ""
    tokens = [token.strip() for token in str(value).split("|") if token.strip()]
    return "|".join(sorted(dict.fromkeys(tokens)))


def load_alibaba_pods(raw_dir: str | Path = DEFAULT_RAW_DIR) -> pd.DataFrame:
    """Load the Alibaba pod trace used for generator calibration."""

    path = Path(raw_dir) / POD_TRACE_FILE
    if not path.exists():
        raise FileNotFoundError(f"Alibaba pod trace not found: {path}")
    return pd.read_csv(path)


def prepare_alibaba_generator_samples(
    pods_df: pd.DataFrame,
    *,
    max_runtime_hours: float = 4.0,
    slot_minutes: int = 60,
) -> pd.DataFrame:
    """Return filtered, model-ready empirical job samples from Alibaba pods."""

    if max_runtime_hours <= 0:
        raise ValueError("max_runtime_hours must be positive")
    if slot_minutes <= 0:
        raise ValueError("slot_minutes must be positive")

    samples = pods_df.copy()
    samples["gpu_type_required"] = samples["gpu_spec"].map(normalize_gpu_spec)
    samples["runtime_hours"] = (samples["deletion_time"] - samples["scheduled_time"]) / 3600.0
    samples["wait_hours"] = (samples["scheduled_time"] - samples["creation_time"]).clip(lower=0) / 3600.0
    samples["cpu_required"] = samples["cpu_milli"].astype(float) / 1000.0
    samples["memory_required_gb"] = samples["memory_mib"].astype(float) / 1024.0
    samples["effective_gpu"] = np.maximum(
        samples["num_gpu"].astype(float),
        np.ceil(samples["gpu_milli"].astype(float) / 1000.0),
    )
    samples["gpu_count_required"] = samples["effective_gpu"].astype(int).clip(lower=1)
    samples["duration"] = np.ceil(samples["runtime_hours"] * 60.0 / slot_minutes).astype("Int64")
    samples["slot_minutes"] = int(slot_minutes)

    valid = (
        (samples["num_gpu"].astype(float) > 0)
        & (samples["gpu_milli"].astype(float) > 0)
        & samples["scheduled_time"].notna()
        & (samples["deletion_time"] > samples["scheduled_time"])
        & (samples["runtime_hours"] > 0)
        & (samples["runtime_hours"] <= max_runtime_hours)
        & (samples["cpu_required"] > 0)
        & (samples["memory_required_gb"] > 0)
        & samples["duration"].notna()
    )
    columns = [
        "name",
        "pod_phase",
        "qos",
        "gpu_type_required",
        "gpu_count_required",
        "cpu_required",
        "memory_required_gb",
        "runtime_hours",
        "duration",
        "slot_minutes",
        "wait_hours",
        "creation_time",
        "scheduled_time",
        "deletion_time",
    ]
    return samples.loc[valid, columns].reset_index(drop=True)


def summarize_jobs_per_24h(pods_df: pd.DataFrame, samples_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize observed GPU job arrivals per 24-hour window."""

    gpu_jobs = pods_df[pods_df["num_gpu"].astype(float) > 0].copy()
    if gpu_jobs.empty:
        raise ValueError("Alibaba pod trace contains no GPU jobs")

    min_creation = float(gpu_jobs["creation_time"].min())
    max_creation = float(gpu_jobs["creation_time"].max())
    trace_days = max((max_creation - min_creation) / SECONDS_PER_DAY, 1.0 / 24.0)

    gpu_jobs["creation_day"] = np.floor((gpu_jobs["creation_time"] - min_creation) / SECONDS_PER_DAY).astype(int)
    samples = samples_df.copy()
    samples["creation_day"] = np.floor((samples["creation_time"] - min_creation) / SECONDS_PER_DAY).astype(int)

    all_daily = gpu_jobs.groupby("creation_day").size()
    filtered_daily = samples.groupby("creation_day").size()
    daily = pd.DataFrame(
        {
            "creation_day": sorted(set(all_daily.index).union(filtered_daily.index)),
        }
    )
    daily["all_gpu_jobs"] = daily["creation_day"].map(all_daily).fillna(0).astype(int)
    daily["filtered_generator_jobs"] = daily["creation_day"].map(filtered_daily).fillna(0).astype(int)

    summary = pd.DataFrame(
        [
            {
                "metric": "trace_span_days",
                "value": trace_days,
            },
            {
                "metric": "all_gpu_jobs_per_24h_mean_by_day",
                "value": float(daily["all_gpu_jobs"].mean()),
            },
            {
                "metric": "all_gpu_jobs_per_24h_rate_over_span",
                "value": float(len(gpu_jobs) / trace_days),
            },
            {
                "metric": "filtered_jobs_per_24h_mean_by_day",
                "value": float(daily["filtered_generator_jobs"].mean()),
            },
            {
                "metric": "filtered_jobs_per_24h_rate_over_span",
                "value": float(len(samples_df) / trace_days),
            },
        ]
    )
    return daily, summary


def build_alibaba_generator_calibration(config: AlibabaCalibrationConfig) -> dict[str, pd.DataFrame]:
    """Build and write generator calibration tables from local Alibaba traces."""

    pods = load_alibaba_pods(config.raw_dir)
    samples = prepare_alibaba_generator_samples(
        pods,
        max_runtime_hours=config.max_runtime_hours,
        slot_minutes=config.slot_minutes,
    )
    if samples.empty:
        raise ValueError("no Alibaba jobs remain after generator calibration filters")

    compatibility = (
        samples.assign(
            gpu_type_label=lambda df: df["gpu_type_required"].replace("", "NO_GPU_TYPE_CONSTRAINT")
        )
        .groupby("gpu_type_label", dropna=False)
        .size()
        .reset_index(name="jobs")
        .sort_values("jobs", ascending=False)
        .reset_index(drop=True)
    )
    compatibility["probability"] = compatibility["jobs"] / compatibility["jobs"].sum()

    duration = (
        samples.groupby("duration", dropna=False)
        .size()
        .reset_index(name="jobs")
        .sort_values("duration")
        .reset_index(drop=True)
    )
    duration["probability"] = duration["jobs"] / duration["jobs"].sum()

    gpu_count = (
        samples.groupby("gpu_count_required", dropna=False)
        .size()
        .reset_index(name="jobs")
        .sort_values("gpu_count_required")
        .reset_index(drop=True)
    )
    gpu_count["probability"] = gpu_count["jobs"] / gpu_count["jobs"].sum()

    resource_quantiles = samples[
        ["gpu_count_required", "cpu_required", "memory_required_gb", "runtime_hours", "duration", "wait_hours"]
    ].quantile([0.1, 0.25, 0.5, 0.75, 0.9, 0.95]).reset_index(names="quantile")

    daily_jobs, jobs_per_24h = summarize_jobs_per_24h(pods, samples)

    summary = pd.DataFrame(
        [
            {"metric": "raw_pods", "value": float(len(pods))},
            {"metric": "raw_gpu_pods", "value": float((pods["num_gpu"].astype(float) > 0).sum())},
            {"metric": "generator_sample_jobs", "value": float(len(samples))},
            {"metric": "max_runtime_hours_filter", "value": float(config.max_runtime_hours)},
            {"metric": "slot_minutes", "value": float(config.slot_minutes)},
            {
                "metric": "unrestricted_gpu_compatibility_share",
                "value": float((samples["gpu_type_required"] == "").mean()),
            },
            {"metric": "median_runtime_hours", "value": float(samples["runtime_hours"].median())},
            {"metric": "median_duration_slots", "value": float(samples["duration"].median())},
            {"metric": "median_gpu_count_required", "value": float(samples["gpu_count_required"].median())},
        ]
    )
    summary = pd.concat([summary, jobs_per_24h], ignore_index=True)

    config.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "alibaba_generator_job_samples": samples,
        "alibaba_gpu_compatibility_distribution": compatibility,
        "alibaba_duration_distribution_slots": duration,
        "alibaba_gpu_count_distribution": gpu_count,
        "alibaba_resource_quantiles": resource_quantiles,
        "alibaba_jobs_per_24h": daily_jobs,
        "alibaba_generator_calibration_summary": summary,
    }
    for name, frame in outputs.items():
        frame.to_csv(config.output_dir / f"{name}.csv", index=False)
    return outputs
