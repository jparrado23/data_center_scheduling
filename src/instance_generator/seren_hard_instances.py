"""Construct planted-feasible Seren benchmarks with controlled MILP hardness.

The historical reduced benchmarks used tiny GPU-type pools.  This module
translates their *structural* difficulty drivers to Seren's A100 node model:
resource utilisation, demand competing for a cheap time block, partial
job-to-partition compatibility, multi-node packing, and flexible windows.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SerenHardnessConfig:
    """Parameters for one planted-feasible Seren hardness benchmark."""

    num_jobs: int = 90
    pressure_start: int = 6
    pressure_end: int = 18
    cheap_core_start: int = 10
    cheap_core_length: int = 4
    target_node_utilization: float = 0.85
    utilization_tolerance: float = 0.05
    cheap_block_pressure: float = 1.70
    cheap_pressure_tolerance: float = 0.20
    restricted_compatibility_share: float = 0.50
    window_min_options: int = 4
    window_max_options: int = 10
    max_duration_hours: int = 5
    synthetic_duration_values: tuple[int, ...] | None = None
    synthetic_duration_probabilities: tuple[float, ...] | None = None
    max_attempts: int = 5_000

    def __post_init__(self) -> None:
        """Validate configuration bounds before generating an instance."""

        if self.num_jobs <= 0:
            raise ValueError("num_jobs must be positive")
        if self.pressure_end <= self.pressure_start:
            raise ValueError("pressure window must have positive length")
        if self.cheap_core_length <= 0:
            raise ValueError("cheap_core_length must be positive")
        if not self.pressure_start <= self.cheap_core_start:
            raise ValueError("cheap core must start inside the pressure window")
        if self.cheap_core_end > self.pressure_end:
            raise ValueError("cheap core must end inside the pressure window")
        if not 0 < self.target_node_utilization <= 1:
            raise ValueError("target_node_utilization must be in (0, 1]")
        if self.utilization_tolerance < 0:
            raise ValueError("utilization_tolerance cannot be negative")
        if self.cheap_block_pressure <= 0 or self.cheap_pressure_tolerance < 0:
            raise ValueError("cheap pressure values must be non-negative")
        if not 0 <= self.restricted_compatibility_share <= 1:
            raise ValueError("restricted_compatibility_share must be in [0, 1]")
        if not 2 <= self.window_min_options <= self.window_max_options:
            raise ValueError("window options must satisfy 2 <= min <= max")
        if self.max_duration_hours <= 0 or self.max_attempts <= 0:
            raise ValueError("duration and max_attempts must be positive")
        if (self.synthetic_duration_values is None) != (self.synthetic_duration_probabilities is None):
            raise ValueError("synthetic duration values and probabilities must be supplied together")
        if self.synthetic_duration_values is not None:
            if len(self.synthetic_duration_values) == 0:
                raise ValueError("synthetic_duration_values cannot be empty")
            if len(self.synthetic_duration_values) != len(self.synthetic_duration_probabilities):
                raise ValueError("synthetic duration values and probabilities must have equal length")
            if any(value <= 0 or value > self.max_duration_hours for value in self.synthetic_duration_values):
                raise ValueError("synthetic durations must be positive and within max_duration_hours")
            if any(value < 0 for value in self.synthetic_duration_probabilities) or not np.isclose(
                sum(self.synthetic_duration_probabilities), 1.0
            ):
                raise ValueError("synthetic duration probabilities must be non-negative and sum to one")

    @property
    def pressure_length(self) -> int:
        """Return the number of slots in the construction pressure window."""

        return self.pressure_end - self.pressure_start

    @property
    def cheap_core_end(self) -> int:
        """Return the exclusive end of the cheap-core interval."""

        return self.cheap_core_start + self.cheap_core_length


def build_logical_a100_partitions(
    source_clusters: pd.DataFrame,
    capacities: dict[str, int],
) -> pd.DataFrame:
    """Split Seren's A100 pool into explicit scheduling partitions in node units."""

    if sum(capacities.values()) != int(source_clusters["node_capacity"].sum()):
        raise ValueError("Partition capacities must sum to Seren's node capacity")
    template = source_clusters.iloc[0].to_dict()
    rows: list[dict[str, Any]] = []
    for cluster_id, nodes in capacities.items():
        row = template.copy()
        raw_gpus = int(nodes) * 8
        row.update(
            {
                "cluster_id": cluster_id,
                "cluster_role": "logical_a100_partition",
                "gpu_type": "NVIDIA A100-SXM 80GB",
                "node_capacity": int(nodes),
                "gpu_count": raw_gpus,
                "gpu_capacity": raw_gpus,
                "raw_gpu_count": raw_gpus,
                "block_capacity": raw_gpus // 96,
                "capacity_kw": raw_gpus * 0.4,
                "power_capacity_kw": raw_gpus * 0.4,
                "capacity": raw_gpus * 0.4 / 1000.0,
                "cpu_capacity": int(nodes) * 128,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _overlaps_core(start: int, duration: int, config: SerenHardnessConfig) -> bool:
    """Return whether a start choice makes a job active in the cheap core."""

    return start < config.cheap_core_end and start + duration > config.cheap_core_start


def _balanced_hidden_placement(
    jobs: pd.DataFrame,
    capacities: dict[str, int],
    cheap_candidates: set[int],
    config: SerenHardnessConfig,
    rng: np.random.Generator,
) -> tuple[dict[int, tuple[str, int]], dict[str, np.ndarray]] | None:
    """Place jobs once while preserving feasibility and cheap-core eligibility.

    Jobs selected as cheap candidates may be placed anywhere in the pressure
    window; their exposed window will include a core-overlapping alternative.
    Other jobs are planted outside the core so that the pressure metric is not
    inflated by every job in the instance.
    """

    loads = {
        cluster_id: np.zeros(config.pressure_length, dtype=int)
        for cluster_id in capacities
    }
    placements: dict[int, tuple[str, int]] = {}
    order = jobs.assign(
        node_hours=lambda frame: frame["node_units_required"] * frame["duration"]
    ).sort_values("node_hours", ascending=False).index

    for index in order:
        row = jobs.loc[index]
        nodes = int(row["node_units_required"])
        duration = int(row["duration"])
        candidates: list[tuple[float, float, str, int, np.ndarray]] = []
        for cluster_id, capacity in capacities.items():
            if nodes > capacity:
                continue
            for start in range(config.pressure_start, config.pressure_end - duration + 1):
                if index not in cheap_candidates and _overlaps_core(start, duration, config):
                    continue
                offsets = np.arange(
                    start - config.pressure_start,
                    start - config.pressure_start + duration,
                )
                proposed = loads[cluster_id].copy()
                proposed[offsets] += nodes
                if proposed.max() > capacity:
                    continue
                candidates.append(
                    (
                        float(proposed.max() / capacity),
                        float(proposed.sum() / (capacity * config.pressure_length)),
                        cluster_id,
                        start,
                        proposed,
                    )
                )
        if not candidates:
            return None
        best_peak = min(candidate[0] for candidate in candidates)
        peak_ties = [candidate for candidate in candidates if candidate[0] == best_peak]
        best_load = min(candidate[1] for candidate in peak_ties)
        tied = [candidate for candidate in peak_ties if candidate[1] == best_load]
        _, _, cluster_id, start, selected = tied[int(rng.integers(len(tied)))]
        loads[cluster_id] = selected
        placements[int(index)] = (cluster_id, int(start))
    return placements, loads


def _window_with_core_option(
    hidden_start: int,
    duration: int,
    config: SerenHardnessConfig,
    rng: np.random.Generator,
) -> tuple[int, int] | None:
    """Expose a 4--10-option window containing the placement and cheap core."""

    core_starts = [
        start
        for start in range(0, 25 - duration)
        if _overlaps_core(start, duration, config)
    ]
    possible = [
        start for start in core_starts if abs(start - hidden_start) + 1 <= config.window_max_options
    ]
    if not possible:
        return None
    core_start = possible[int(rng.integers(len(possible)))]
    minimum = max(config.window_min_options, abs(core_start - hidden_start) + 1)
    if minimum > config.window_max_options:
        return None
    options = int(rng.integers(minimum, config.window_max_options + 1))
    lower = max(0, max(hidden_start, core_start) - options + 1)
    upper = min(min(hidden_start, core_start), 24 - duration - options + 1)
    if lower > upper:
        return None
    earliest = int(rng.integers(lower, upper + 1))
    return earliest, earliest + options - 1


def _window_without_core_option(
    hidden_start: int,
    duration: int,
    config: SerenHardnessConfig,
    rng: np.random.Generator,
) -> tuple[int, int] | None:
    """Expose a flexible window containing the placement but avoiding the core."""

    candidates: list[tuple[int, int]] = []
    for options in range(config.window_min_options, config.window_max_options + 1):
        for earliest in range(0, 25 - duration - options + 1):
            latest = earliest + options - 1
            if earliest <= hidden_start <= latest and not any(
                _overlaps_core(start, duration, config)
                for start in range(earliest, latest + 1)
            ):
                candidates.append((earliest, latest))
    return candidates[int(rng.integers(len(candidates)))] if candidates else None


def _compatible_partitions(
    planted_cluster: str,
    nodes: int,
    capacities: dict[str, int],
    restrict: bool,
) -> list[str]:
    """Return explicit cluster compatibility while always retaining the placement."""

    feasible = [cluster for cluster, capacity in capacities.items() if nodes <= capacity]
    if planted_cluster not in feasible:
        raise ValueError("Planted cluster cannot host the selected job")
    return [planted_cluster] if restrict else feasible


def generate_seren_hard_instance(
    source_jobs: pd.DataFrame,
    source_clusters: pd.DataFrame,
    capacities: dict[str, int],
    config: SerenHardnessConfig,
    seed: int,
    instance_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Generate a feasible Seren instance carrying the historical hardness traits.

    `source_jobs` supplies empirical duration and node-demand samples.  The
    hidden schedule is persisted only as a feasibility certificate; it is not
    an input to the optimiser.
    """

    rng = np.random.default_rng(seed)
    clusters = build_logical_a100_partitions(source_clusters, capacities)
    source = source_jobs[
        (source_jobs["duration"] <= config.max_duration_hours)
        & (source_jobs["node_units_required"] <= max(capacities.values()))
    ].reset_index(drop=True)
    if source.empty:
        raise ValueError("No empirical source jobs fit the configured benchmark")

    capacity = sum(capacities.values())
    total_capacity_hours = capacity * config.pressure_length
    cheap_capacity_hours = capacity * config.cheap_core_length
    for attempt in range(1, config.max_attempts + 1):
        sampled = source.iloc[rng.integers(0, len(source), size=config.num_jobs)].copy().reset_index(drop=True)
        sampled["job_id"] = [f"{instance_name}_{index:03d}" for index in range(config.num_jobs)]
        if config.synthetic_duration_values is not None:
            sampled["duration_sample_origin"] = "synthetic_stress_profile"
            sampled["duration"] = rng.choice(
                config.synthetic_duration_values,
                size=config.num_jobs,
                p=config.synthetic_duration_probabilities,
            ).astype(int)
        node_hours = sampled["node_units_required"] * sampled["duration"]
        utilization = float(node_hours.sum() / total_capacity_hours)
        if abs(utilization - config.target_node_utilization) > config.utilization_tolerance:
            continue

        target_cheap_hours = config.cheap_block_pressure * cheap_capacity_hours
        shuffled = list(sampled.index)
        rng.shuffle(shuffled)
        cheap_candidates: set[int] = set()
        cheap_hours = 0.0
        for index in shuffled:
            if cheap_hours < target_cheap_hours:
                cheap_candidates.add(int(index))
                cheap_hours += float(node_hours.loc[index])
        cheap_pressure = cheap_hours / cheap_capacity_hours
        if abs(cheap_pressure - config.cheap_block_pressure) > config.cheap_pressure_tolerance:
            continue

        packed = _balanced_hidden_placement(sampled, capacities, cheap_candidates, config, rng)
        if packed is None:
            continue
        placements, loads = packed
        restricted = set(rng.choice(
            sampled.index.to_numpy(),
            size=round(config.restricted_compatibility_share * config.num_jobs),
            replace=False,
        ).tolist())
        valid = True
        rows: list[dict[str, Any]] = []
        for index, row in sampled.iterrows():
            planted_cluster, planted_start = placements[int(index)]
            duration = int(row["duration"])
            window = (
                _window_with_core_option(planted_start, duration, config, rng)
                if int(index) in cheap_candidates
                else _window_without_core_option(planted_start, duration, config, rng)
            )
            if window is None:
                valid = False
                break
            earliest, latest = window
            updated = row.to_dict()
            updated.update(
                {
                    "received_hour": earliest,
                    "earliest_start": earliest,
                    "latest_start": latest,
                    "compatible_clusters": "|".join(
                        _compatible_partitions(
                            planted_cluster,
                            int(row["node_units_required"]),
                            capacities,
                            int(index) in restricted,
                        )
                    ),
                }
            )
            rows.append(updated)
        if not valid:
            continue

        jobs = pd.DataFrame(rows)
        planted = jobs[["job_id", "node_units_required", "duration"]].copy()
        planted["planted_cluster"] = planted.index.map(lambda index: placements[int(index)][0])
        planted["planted_start"] = planted.index.map(lambda index: placements[int(index)][1])
        actual_cheap_hours = float(
            (jobs.loc[
                jobs.apply(
                    lambda row: any(
                        _overlaps_core(start, int(row["duration"]), config)
                        for start in range(int(row["earliest_start"]), int(row["latest_start"]) + 1)
                    ),
                    axis=1,
                ),
                "node_units_required",
            ] * jobs.loc[
                jobs.apply(
                    lambda row: any(
                        _overlaps_core(start, int(row["duration"]), config)
                        for start in range(int(row["earliest_start"]), int(row["latest_start"]) + 1)
                    ),
                    axis=1,
                ),
                "duration",
            ]).sum()
        )
        metadata = {
            "generator": "seren_legacy_shaped_combinatorial_hardness",
            "seed": seed,
            "attempt": attempt,
            "instance_name": instance_name,
            "config": asdict(config),
            "pressure_node_hour_utilization": utilization,
            "cheap_block_pressure_actual": actual_cheap_hours / cheap_capacity_hours,
            "planted_peak_node_utilization": float(max(
                load.max() / capacities[cluster_id] for cluster_id, load in loads.items()
            )),
            "restricted_compatibility_share_actual": float(
                (jobs["compatible_clusters"].str.split("|").str.len() == 1).mean()
            ),
            "mean_compatible_clusters": float(jobs["compatible_clusters"].str.split("|").str.len().mean()),
            "mean_duration_hours": float(jobs["duration"].mean()),
            "mean_node_demand": float(jobs["node_units_required"].mean()),
            "mean_start_options": float((jobs["latest_start"] - jobs["earliest_start"] + 1).mean()),
        }
        return jobs, clusters, planted, metadata
    raise RuntimeError("Could not construct a planted-feasible Seren hardness instance")
