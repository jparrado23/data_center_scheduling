"""Generate small optimization-hard benchmark instances.

These instances are designed to stress the MILP search tree without relying on
very large job counts. The generator creates a hidden feasible schedule, then
exposes wider job windows with contention around cheap renewable slots.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import ModelConfig
from src.data.scenarios import DEFAULT_GPU_POWER_KW
from src.data.validation import validate_clusters, validate_hourly_inputs, validate_jobs


DEFAULT_SMALL_GPU_COUNTS = {"G2": 4, "T4": 4, "V100M32": 2, "P100": 2}


@dataclass(frozen=True)
class DifficultyConfig:
    """Controls one reduced, feasible, optimization-hard instance.

    The main stress knobs are global GPU-slot utilization, cheap-window
    contention, restricted GPU-type compatibility, and multi-GPU packing.
    """

    target_gpu_utilization: float = 0.70
    cheap_block_pressure: float = 1.40
    cheap_window_share: float = 0.75
    restricted_gpu_share: float = 0.35
    multi_gpu_share: float = 0.30
    window_min: int = 4
    window_max: int = 10
    cheap_block_start: int = 5
    cheap_block_length: int = 5
    max_jobs: int = 60
    horizon_slots: int = 16
    slot_minutes: int = 15
    price_contrast: float = 1.50
    contracted_power_ratio: float = 0.70
    cpu_per_gpu: float = 8.0
    memory_gb_per_gpu: float = 48.0
    cpu_jitter: float = 0.20
    memory_jitter: float = 0.20
    power_jitter: float = 0.10
    pue: float = 1.20
    renewable_price: float = 35.0
    peak_price: float = 800.0
    baseline_load_mw: float = 0.0
    max_generation_attempts: int = 20_000

    def __post_init__(self) -> None:
        """Validate physical and sampling parameters."""

        if not 0 < self.target_gpu_utilization <= 1:
            raise ValueError("target_gpu_utilization must be in (0, 1]")
        if self.cheap_block_pressure <= 0:
            raise ValueError("cheap_block_pressure must be positive")
        for name in ("cheap_window_share", "restricted_gpu_share", "multi_gpu_share"):
            value = getattr(self, name)
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be between 0 and 1")
        if self.window_min <= 0 or self.window_max < self.window_min:
            raise ValueError("window bounds must be positive and ordered")
        if self.cheap_block_start < 0 or self.cheap_block_length <= 0:
            raise ValueError("cheap block start and length must define a positive block")
        if self.cheap_block_end > self.horizon_slots:
            raise ValueError("cheap block must fit within horizon_slots")
        if self.max_jobs <= 0:
            raise ValueError("max_jobs must be positive")
        if self.horizon_slots <= 0:
            raise ValueError("horizon_slots must be positive")
        if self.slot_minutes <= 0:
            raise ValueError("slot_minutes must be positive")
        if self.price_contrast <= 0:
            raise ValueError("price_contrast must be positive")
        if self.contracted_power_ratio <= 0:
            raise ValueError("contracted_power_ratio must be positive")
        if self.cpu_per_gpu <= 0 or self.memory_gb_per_gpu <= 0:
            raise ValueError("CPU and memory per GPU must be positive")
        for name in ("cpu_jitter", "memory_jitter", "power_jitter"):
            value = getattr(self, name)
            if not 0 <= value < 1:
                raise ValueError(f"{name} must be in [0, 1)")
        if self.max_generation_attempts <= 0:
            raise ValueError("max_generation_attempts must be positive")
        if self.pue <= 0:
            raise ValueError("pue must be positive")
        if self.renewable_price < 0 or self.peak_price < 0 or self.baseline_load_mw < 0:
            raise ValueError("prices and baseline load must be non-negative")

    @property
    def cheap_block_end(self) -> int:
        """Return the exclusive end slot of the cheap block."""

        return self.cheap_block_start + self.cheap_block_length

    @property
    def delta_t(self) -> float:
        """Return slot length in hours."""

        return self.slot_minutes / 60.0


@dataclass(frozen=True)
class BenchmarkInstanceSpec:
    """Names a reusable hard-instance configuration and its random seeds."""

    label: str
    config: DifficultyConfig
    seeds: tuple[int, ...]


def build_small_clusters(gpu_counts: dict[str, int] | None = None) -> pd.DataFrame:
    """Build a reduced one-cluster-per-GPU-type table for hard benchmarks."""

    rows = []
    for gpu_type, gpu_count in (gpu_counts or DEFAULT_SMALL_GPU_COUNTS).items():
        if gpu_type not in DEFAULT_GPU_POWER_KW:
            raise ValueError(f"No default GPU power is available for {gpu_type!r}")
        power_capacity_kw = gpu_count * DEFAULT_GPU_POWER_KW[gpu_type]
        rows.append(
            {
                "cluster_id": f"cluster_{gpu_type.lower()}",
                "cluster_role": "small_gpu_type_pool",
                "capacity_kw": power_capacity_kw,
                "capacity": power_capacity_kw / 1000.0,
                "power_capacity_kw": power_capacity_kw,
                "gpu_type": gpu_type,
                "gpu_count": gpu_count,
                "gpu_capacity": gpu_count,
                "cpu_capacity": gpu_count * 8.0,
                "memory_capacity_gb": gpu_count * 48.0,
            }
        )
    clusters = pd.DataFrame(rows)
    validate_clusters(clusters)
    return clusters


def build_energy_inputs(config: DifficultyConfig, clusters: pd.DataFrame) -> tuple[pd.DataFrame, ModelConfig]:
    """Build slot-level energy inputs and scalar MILP config."""

    total_it_capacity = float(clusters["capacity"].sum())
    cheap_slots = set(range(config.cheap_block_start, config.cheap_block_end))
    rows = []
    for slot in range(config.horizon_slots):
        is_cheap = slot in cheap_slots
        grid_price = 70.0 / config.price_contrast if is_cheap else 70.0 * config.price_contrast
        renewable_available = (0.55 if is_cheap else 0.05) * total_it_capacity * config.pue
        rows.append(
            {
                "hour": slot,
                "renewable_available": renewable_available,
                "grid_price": grid_price,
                "baseline_load": config.baseline_load_mw,
                "pue": config.pue,
            }
        )
    hourly = pd.DataFrame(rows)
    validate_hourly_inputs(hourly)
    model_config = ModelConfig(
        contracted_power=config.contracted_power_ratio * total_it_capacity * config.pue,
        renewable_price=config.renewable_price,
        peak_price=config.peak_price,
        delta_t=config.delta_t,
        pue=config.pue,
    )
    return hourly, model_config


def window_intersects_block(earliest: int, latest: int, duration: int, config: DifficultyConfig) -> bool:
    """Return whether at least one feasible start overlaps the cheap block."""

    for start in range(earliest, latest + 1):
        if start < config.cheap_block_end and start + duration > config.cheap_block_start:
            return True
    return False


def current_gpu_hours(jobs: list[dict[str, Any]]) -> int:
    """Return total GPU-slot demand from generated job dictionaries."""

    return int(sum(job["gpu_count_required"] * job["duration"] for job in jobs))


def cheap_window_gpu_hours(jobs: list[dict[str, Any]], config: DifficultyConfig) -> int:
    """Return GPU-slot demand whose windows can overlap the cheap block."""

    return int(
        sum(
            job["gpu_count_required"] * job["duration"]
            for job in jobs
            if window_intersects_block(job["earliest_start"], job["latest_start"], job["duration"], config)
        )
    )


def choose_job_gpu_requirement(
    rng: np.random.Generator,
    config: DifficultyConfig,
    clusters: pd.DataFrame,
) -> tuple[int, str]:
    """Sample GPU count and optional single-GPU-type restriction."""

    gpu_count = int(rng.choice([2, 3], p=[0.75, 0.25])) if rng.random() < config.multi_gpu_share else 1
    feasible_types = clusters.loc[clusters["gpu_count"] >= gpu_count, "gpu_type"].tolist()
    if not feasible_types:
        raise ValueError(f"No cluster can host a job requiring {gpu_count} GPUs")
    gpu_type_required = str(rng.choice(feasible_types)) if rng.random() < config.restricted_gpu_share else ""
    return gpu_count, gpu_type_required


def compatible_clusters_for_job(gpu_count: int, gpu_type_required: str, clusters: pd.DataFrame) -> list[str]:
    """Return cluster ids that can run a sampled GPU requirement."""

    allowed_types = set(clusters["gpu_type"]) if gpu_type_required == "" else {gpu_type_required}
    compatible = clusters[(clusters["gpu_type"].isin(allowed_types)) & (clusters["gpu_count"] >= gpu_count)]
    return compatible["cluster_id"].tolist()


def can_place_hidden(
    cluster: str,
    start: int,
    duration: int,
    gpu_count: int,
    cpu: float,
    memory: float,
    power: float,
    usage: dict[str, dict[str, np.ndarray]],
    clusters_by_id: dict[str, dict[str, Any]],
) -> bool:
    """Check whether a hidden placement respects all resource capacities."""

    end = start + duration
    cluster_data = clusters_by_id[cluster]
    return bool(
        np.all(usage[cluster]["gpu"][start:end] + gpu_count <= cluster_data["gpu_count"])
        and np.all(usage[cluster]["cpu"][start:end] + cpu <= cluster_data["cpu_capacity"])
        and np.all(usage[cluster]["memory"][start:end] + memory <= cluster_data["memory_capacity_gb"])
        and np.all(usage[cluster]["power"][start:end] + power <= cluster_data["capacity"])
    )


def reserve_hidden(
    cluster: str,
    start: int,
    duration: int,
    gpu_count: int,
    cpu: float,
    memory: float,
    power: float,
    usage: dict[str, dict[str, np.ndarray]],
) -> None:
    """Reserve resources after accepting one hidden feasible placement."""

    end = start + duration
    usage[cluster]["gpu"][start:end] += gpu_count
    usage[cluster]["cpu"][start:end] += cpu
    usage[cluster]["memory"][start:end] += memory
    usage[cluster]["power"][start:end] += power


def make_window(
    hidden_start: int,
    duration: int,
    should_overlap_cheap: bool,
    rng: np.random.Generator,
    config: DifficultyConfig,
) -> tuple[int, int]:
    """Build a flexible window that includes the hidden feasible start."""

    latest_possible = config.horizon_slots - duration
    if should_overlap_cheap:
        anchor_left = min(hidden_start, config.cheap_block_start)
        anchor_right = max(hidden_start, config.cheap_block_end - duration)
        earliest = max(0, anchor_left - int(rng.integers(0, 3)))
        latest = min(latest_possible, anchor_right + int(rng.integers(0, 3)))
    else:
        target_width = int(rng.integers(config.window_min, config.window_max + 1))
        left_slack = int(rng.integers(0, target_width))
        earliest = max(0, hidden_start - left_slack)
        latest = min(latest_possible, earliest + target_width - 1)
        if hidden_start > latest:
            shift = hidden_start - latest
            earliest = max(0, earliest + shift)
            latest = min(latest_possible, latest + shift)

    if latest < hidden_start:
        latest = hidden_start
    if earliest > hidden_start:
        earliest = hidden_start

    width = latest - earliest + 1
    if width < config.window_min:
        missing = config.window_min - width
        earliest = max(0, earliest - missing // 2 - missing % 2)
        latest = min(latest_possible, latest + missing // 2)
    if latest < hidden_start:
        latest = hidden_start
    if earliest > hidden_start:
        earliest = hidden_start
    return int(earliest), int(latest)


def generate_difficulty_instance(
    config: DifficultyConfig,
    seed: int,
    clusters: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate one feasible instance with controlled optimization difficulty.

    The returned hidden schedule proves feasibility but should not be passed to
    the optimizer. The optimizer only receives jobs, hourly inputs, clusters,
    and scalar model config.
    """

    rng = np.random.default_rng(seed)
    clusters_by_id = clusters.set_index("cluster_id").to_dict(orient="index")
    gpu_types = [str(value) for value in clusters["gpu_type"].tolist()]
    max_gpu_capacity = int(clusters["gpu_count"].max())
    usage = {
        cluster_id: {
            "gpu": np.zeros(config.horizon_slots),
            "cpu": np.zeros(config.horizon_slots),
            "memory": np.zeros(config.horizon_slots),
            "power": np.zeros(config.horizon_slots),
        }
        for cluster_id in clusters_by_id
    }

    total_gpu_capacity_slots = int(clusters["gpu_count"].sum() * config.horizon_slots)
    cheap_gpu_capacity_slots = int(clusters["gpu_count"].sum() * config.cheap_block_length)
    target_gpu_slots = int(np.ceil(config.target_gpu_utilization * total_gpu_capacity_slots))
    target_cheap_gpu_slots = int(np.ceil(config.cheap_block_pressure * cheap_gpu_capacity_slots))

    jobs: list[dict[str, Any]] = []
    hidden_rows: list[dict[str, Any]] = []
    attempts = 0
    current_gpu_slots = 0
    current_cheap_gpu_slots = 0

    while (
        (current_gpu_slots < target_gpu_slots or current_cheap_gpu_slots < target_cheap_gpu_slots)
        and len(jobs) < config.max_jobs
        and attempts < config.max_generation_attempts
    ):
        attempts += 1
        duration = int(rng.choice([1, 2, 3, 4], p=[0.20, 0.35, 0.30, 0.15]))
        if rng.random() < config.multi_gpu_share and max_gpu_capacity >= 2:
            multi_choices = [count for count in (2, 3) if count <= max_gpu_capacity]
            multi_probabilities = [0.75, 0.25][: len(multi_choices)]
            multi_probabilities = np.array(multi_probabilities) / sum(multi_probabilities)
            gpu_count = int(rng.choice(multi_choices, p=multi_probabilities))
        else:
            gpu_count = 1

        feasible_types = [
            str(cluster_data["gpu_type"])
            for cluster_data in clusters_by_id.values()
            if int(cluster_data["gpu_count"]) >= gpu_count
        ]
        gpu_type_required = str(rng.choice(feasible_types)) if rng.random() < config.restricted_gpu_share else ""
        compatible = [
            cluster_id
            for cluster_id, cluster_data in clusters_by_id.items()
            if int(cluster_data["gpu_count"]) >= gpu_count
            and (not gpu_type_required or str(cluster_data["gpu_type"]) == gpu_type_required)
        ]
        if not compatible:
            continue

        per_gpu_cpu = config.cpu_per_gpu * rng.uniform(1.0 - config.cpu_jitter, 1.0 + config.cpu_jitter)
        per_gpu_memory = config.memory_gb_per_gpu * rng.uniform(
            1.0 - config.memory_jitter,
            1.0 + config.memory_jitter,
        )
        representative_gpu_type = gpu_type_required or str(rng.choice(gpu_types))
        per_gpu_power_kw = DEFAULT_GPU_POWER_KW[representative_gpu_type] * rng.uniform(
            1.0 - config.power_jitter,
            1.0 + config.power_jitter,
        )
        cpu = float(gpu_count * per_gpu_cpu)
        memory = float(gpu_count * per_gpu_memory)
        power = float(gpu_count * per_gpu_power_kw / 1000.0)

        cheap_shortfall = current_cheap_gpu_slots < target_cheap_gpu_slots
        should_overlap_cheap = rng.random() < config.cheap_window_share or cheap_shortfall
        if should_overlap_cheap:
            start_low = max(0, config.cheap_block_start - config.window_max)
            start_high = min(config.horizon_slots - duration, config.cheap_block_end + config.window_max)
        else:
            start_low = 0
            start_high = config.horizon_slots - duration
        if start_high < start_low:
            continue

        candidate_clusters = list(compatible)
        rng.shuffle(candidate_clusters)
        candidate_starts = list(range(start_low, start_high + 1))
        rng.shuffle(candidate_starts)

        placed = False
        for cluster in candidate_clusters:
            for hidden_start in candidate_starts:
                if can_place_hidden(cluster, hidden_start, duration, gpu_count, cpu, memory, power, usage, clusters_by_id):
                    earliest, latest = make_window(hidden_start, duration, should_overlap_cheap, rng, config)
                    job_id = f"job_{len(jobs):03d}"
                    jobs.append(
                        {
                            "job_id": job_id,
                            "category": "gpu_restricted" if gpu_type_required else "gpu_unrestricted",
                            "workload_family": "combinatorial_difficulty",
                            "duration": duration,
                            "power": power,
                            "earliest_start": earliest,
                            "latest_start": latest,
                            "gpu_type_required": gpu_type_required,
                            "gpu_count_required": gpu_count,
                            "cpu_required": cpu,
                            "memory_required_gb": memory,
                        }
                    )
                    hidden_rows.append(
                        {
                            "job_id": job_id,
                            "hidden_cluster": cluster,
                            "hidden_start": hidden_start,
                            "duration": duration,
                        }
                    )
                    reserve_hidden(cluster, hidden_start, duration, gpu_count, cpu, memory, power, usage)
                    job_gpu_slots = gpu_count * duration
                    current_gpu_slots += job_gpu_slots
                    if window_intersects_block(earliest, latest, duration, config):
                        current_cheap_gpu_slots += job_gpu_slots
                    placed = True
                    break
            if placed:
                break

    jobs_df = pd.DataFrame(jobs)
    hidden_df = pd.DataFrame(hidden_rows)
    validate_jobs(jobs_df)
    return jobs_df, hidden_df


def instance_diagnostics(
    jobs_df: pd.DataFrame,
    config: DifficultyConfig,
    clusters: pd.DataFrame,
) -> dict[str, Any]:
    """Compute pre-solve structural difficulty metrics."""

    total_gpu_capacity_slots = float(clusters["gpu_count"].sum() * config.horizon_slots)
    cheap_gpu_capacity_slots = float(clusters["gpu_count"].sum() * config.cheap_block_length)
    job_gpu_slots = float((jobs_df["gpu_count_required"] * jobs_df["duration"]).sum())
    cheap_overlap_gpu_slots = float(cheap_window_gpu_hours(jobs_df.to_dict(orient="records"), config))
    window_width = jobs_df["latest_start"] - jobs_df["earliest_start"] + 1
    global_utilization = job_gpu_slots / total_gpu_capacity_slots
    cheap_pressure = cheap_overlap_gpu_slots / cheap_gpu_capacity_slots
    return {
        "num_jobs": int(len(jobs_df)),
        "job_gpu_slots": job_gpu_slots,
        "global_gpu_slot_utilization": global_utilization,
        "cheap_block_pressure_actual": cheap_pressure,
        "global_utilization_target_met": bool(global_utilization >= config.target_gpu_utilization),
        "cheap_pressure_target_met": bool(cheap_pressure >= config.cheap_block_pressure),
        "restricted_gpu_share_actual": float((jobs_df["gpu_type_required"].astype(str).str.len() > 0).mean()),
        "multi_gpu_share_actual": float((jobs_df["gpu_count_required"] > 1).mean()),
        "mean_duration_slots": float(jobs_df["duration"].mean()),
        "median_window_width_slots": float(window_width.median()),
        "mean_window_width_slots": float(window_width.mean()),
        "assignment_density_proxy": float((window_width * jobs_df["gpu_count_required"]).sum()),
    }


def solved_cheap_block_utilization(cluster_results: pd.DataFrame, config: DifficultyConfig) -> float:
    """Return aggregate solved GPU utilization inside the cheap block."""

    block = cluster_results[
        (cluster_results["hour"] >= config.cheap_block_start)
        & (cluster_results["hour"] < config.cheap_block_end)
    ]
    if block.empty or "cluster_gpu_load" not in block.columns:
        return 0.0
    return float(block["cluster_gpu_load"].sum() / block["gpu_capacity"].sum())


def summarize_cluster_utilization(cluster_results: pd.DataFrame) -> dict[str, float]:
    """Summarize power, GPU, CPU, and memory utilization across all slots."""

    metrics: dict[str, float] = {}
    specs = {
        "power": ("cluster_load", "capacity"),
        "gpu": ("cluster_gpu_load", "gpu_capacity"),
        "cpu": ("cluster_cpu_load", "cpu_capacity"),
        "memory": ("cluster_memory_load", "memory_capacity_gb"),
    }
    for label, (load_col, capacity_col) in specs.items():
        if load_col not in cluster_results.columns or capacity_col not in cluster_results.columns:
            continue
        utilization = (cluster_results[load_col] / cluster_results[capacity_col].replace(0, np.nan)).fillna(0.0)
        metrics[f"max_{label}_utilization"] = float(utilization.max())
        metrics[f"mean_{label}_utilization"] = float(utilization.mean())
    return metrics


def safe_json_value(value: Any) -> Any:
    """Convert numpy and pandas scalars into JSON-serializable values."""

    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if value is pd.NA:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def write_hard_instance(
    output_dir: Path,
    label: str,
    config: DifficultyConfig,
    seed: int,
    clusters: pd.DataFrame,
    instance_index: int = 0,
) -> dict[str, Any]:
    """Generate and persist one hard benchmark instance directory."""

    instance_name = f"{instance_index:03d}_{label}_seed_{seed}"
    instance_dir = output_dir / instance_name
    instance_dir.mkdir(parents=True, exist_ok=True)

    jobs_df, hidden_df = generate_difficulty_instance(config, seed, clusters)
    hourly_df, model_config = build_energy_inputs(config, clusters)
    diagnostics = instance_diagnostics(jobs_df, config, clusters)

    jobs_df.to_csv(instance_dir / "jobs.csv", index=False)
    hourly_df.to_csv(instance_dir / "hourly.csv", index=False)
    clusters.to_csv(instance_dir / "clusters.csv", index=False)
    hidden_df.to_csv(instance_dir / "hidden_feasible_schedule.csv", index=False)

    metadata = {
        "instance_name": instance_name,
        "label": label,
        "seed": seed,
        "slot_minutes": config.slot_minutes,
        "horizon_slots": config.horizon_slots,
        "delta_t": config.delta_t,
        "difficulty_config": asdict(config),
        "model_config": asdict(model_config),
        "diagnostics": diagnostics,
        "files": {
            "jobs": "jobs.csv",
            "hourly": "hourly.csv",
            "clusters": "clusters.csv",
            "hidden_feasible_schedule": "hidden_feasible_schedule.csv",
        },
    }
    metadata = json.loads(json.dumps(metadata, default=safe_json_value))
    (instance_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    return {
        "instance_name": instance_name,
        "instance_dir": str(instance_dir),
        "label": label,
        "seed": seed,
        **diagnostics,
    }


def default_hard_benchmark_specs(seeds: tuple[int, ...] = (91_001, 91_002, 91_003)) -> list[BenchmarkInstanceSpec]:
    """Return the curated hard-region specs used for reusable benchmarks."""

    def offset_seeds(offset: int) -> tuple[int, ...]:
        """Return family-specific seeds to avoid duplicate generated jobs."""

        return tuple(seed + offset for seed in seeds)

    return [
        BenchmarkInstanceSpec(
            "high_util_restricted_multi03_pressure14",
            DifficultyConfig(
                target_gpu_utilization=0.90,
                cheap_block_pressure=1.40,
                restricted_gpu_share=0.50,
                multi_gpu_share=0.30,
                max_jobs=90,
            ),
            offset_seeds(0),
        ),
        BenchmarkInstanceSpec(
            "high_util_restricted_multi05_pressure14",
            DifficultyConfig(
                target_gpu_utilization=0.90,
                cheap_block_pressure=1.40,
                restricted_gpu_share=0.50,
                multi_gpu_share=0.50,
                max_jobs=90,
            ),
            offset_seeds(1_000),
        ),
        BenchmarkInstanceSpec(
            "high_util_restricted_multi03_pressure17",
            DifficultyConfig(
                target_gpu_utilization=0.90,
                cheap_block_pressure=1.70,
                restricted_gpu_share=0.50,
                multi_gpu_share=0.30,
                max_jobs=90,
            ),
            offset_seeds(2_000),
        ),
        BenchmarkInstanceSpec(
            "high_util_restricted_multi05_pressure17",
            DifficultyConfig(
                target_gpu_utilization=0.90,
                cheap_block_pressure=1.70,
                restricted_gpu_share=0.50,
                multi_gpu_share=0.50,
                max_jobs=90,
            ),
            offset_seeds(3_000),
        ),
        BenchmarkInstanceSpec(
            "high_util_restricted_multi03_pressure20",
            DifficultyConfig(
                target_gpu_utilization=0.90,
                cheap_block_pressure=2.00,
                restricted_gpu_share=0.50,
                multi_gpu_share=0.30,
                max_jobs=90,
            ),
            offset_seeds(4_000),
        ),
        BenchmarkInstanceSpec(
            "slightly_lower_util_restricted_multi05_pressure20",
            DifficultyConfig(
                target_gpu_utilization=0.85,
                cheap_block_pressure=2.00,
                restricted_gpu_share=0.50,
                multi_gpu_share=0.50,
                max_jobs=90,
            ),
            offset_seeds(5_000),
        ),
    ]


def create_hard_benchmark_pack(
    output_dir: Path,
    specs: list[BenchmarkInstanceSpec] | None = None,
    clusters: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Create a reusable pack of hard benchmark instances and its manifest."""

    output_dir.mkdir(parents=True, exist_ok=True)
    benchmark_specs = specs or default_hard_benchmark_specs()
    clusters_df = build_small_clusters() if clusters is None else clusters

    rows = []
    instance_index = 0
    for spec in benchmark_specs:
        for seed in spec.seeds:
            rows.append(
                write_hard_instance(
                    output_dir=output_dir,
                    label=spec.label,
                    config=spec.config,
                    seed=seed,
                    clusters=clusters_df,
                    instance_index=instance_index,
                )
            )
            instance_index += 1

    manifest = pd.DataFrame(rows)
    manifest.to_csv(output_dir / "manifest.csv", index=False)
    return manifest
