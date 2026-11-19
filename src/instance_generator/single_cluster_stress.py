"""Synthetic energy-tradeoff benchmarks for one A100 node cluster.

The generator deliberately creates feasible scheduling instances with
multi-node jobs and price-sensitive start windows. It does not use workload
categories or cross-cluster compatibility: every job runs on one 256-node
A100 cluster.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd


PLANTING_RULES = frozenset({"greedy_min_peak", "random_feasible"})
DAY_MINUTES = 24 * 60


@dataclass(frozen=True)
class SingleClusterStressConfig:
    """Control the size and structural difficulty of a generated instance."""

    num_jobs: int = 180
    node_capacity: int = 256
    slot_minutes: int = 15
    target_utilization: float = 0.80
    utilization_tolerance: float = 0.05
    node_demand_values: tuple[int, ...] = (8, 16, 32, 64)
    node_demand_probabilities: tuple[float, ...] = (0.45, 0.30, 0.18, 0.07)
    duration_values: tuple[int, ...] = (2, 4, 8, 12)
    duration_probabilities: tuple[float, ...] = (0.25, 0.35, 0.25, 0.15)
    min_start_options: int = 6
    max_start_options: int = 12
    min_price_spread_ratio: float = 1.15
    cheap_price_quantile: float = 0.25
    min_peak_potential_load_ratio: float = 1.50
    pue: float = 1.25
    gpu_tdp_kw: float = 0.4
    gpus_per_node: int = 8
    max_attempts: int = 2_000
    # Difficulty knobs added for the hard-instance frontier search. Every
    # default reproduces the original generator exactly (same RNG stream).
    steer_to_cheap_slot: bool = True
    power_intensity_range: tuple[float, float] = (1.0, 1.0)
    planting: str = "greedy_min_peak"
    random_planting_restarts: int = 25

    def __post_init__(self) -> None:
        """Validate generation parameters."""

        if self.num_jobs <= 0 or self.node_capacity <= 0:
            raise ValueError("num_jobs and node_capacity must be positive")
        if self.slot_minutes <= 0 or 60 % self.slot_minutes != 0:
            raise ValueError("slot_minutes must be a positive divisor of one hour")
        if not 0 < self.target_utilization <= 1 or self.utilization_tolerance < 0:
            raise ValueError("target utilization must be in (0, 1] with non-negative tolerance")
        if not 2 <= self.min_start_options <= self.max_start_options:
            raise ValueError("start options must satisfy 2 <= min <= max")
        if not 1 < self.min_price_spread_ratio:
            raise ValueError("min_price_spread_ratio must exceed one")
        if not 0 < self.cheap_price_quantile < 1:
            raise ValueError("cheap_price_quantile must be in (0, 1)")
        if self.min_peak_potential_load_ratio < 0:
            raise ValueError("min_peak_potential_load_ratio must be non-negative")
        if len(self.power_intensity_range) != 2:
            raise ValueError("power_intensity_range must be a (low, high) pair")
        low, high = self.power_intensity_range
        if not 0 < low <= high:
            raise ValueError("power_intensity_range must satisfy 0 < low <= high")
        if self.planting not in PLANTING_RULES:
            raise ValueError(f"planting must be one of {sorted(PLANTING_RULES)}")
        if self.random_planting_restarts <= 0:
            raise ValueError("random_planting_restarts must be positive")
        if self.pue <= 0 or self.gpu_tdp_kw <= 0 or self.gpus_per_node <= 0:
            raise ValueError("power and PUE parameters must be positive")
        if self.max_attempts <= 0:
            raise ValueError("max_attempts must be positive")
        if len(self.node_demand_values) != len(self.node_demand_probabilities):
            raise ValueError("node demand values and probabilities must have equal length")
        if len(self.duration_values) != len(self.duration_probabilities):
            raise ValueError("duration values and probabilities must have equal length")
        if any(value <= 0 or value > self.node_capacity for value in self.node_demand_values):
            raise ValueError("node demands must be positive and no greater than capacity")
        if any(value <= 0 for value in self.duration_values):
            raise ValueError("durations must be positive")
        for probabilities in (self.node_demand_probabilities, self.duration_probabilities):
            if any(value < 0 for value in probabilities) or not np.isclose(sum(probabilities), 1.0):
                raise ValueError("probabilities must be non-negative and sum to one")

    @property
    def delta_t(self) -> float:
        """Return the slot length in hours."""

        return self.slot_minutes / 60.0

    @property
    def power_per_node_mw(self) -> float:
        """Return maximum IT power of one eight-GPU A100 node in MW."""

        return self.gpus_per_node * self.gpu_tdp_kw / 1000.0


def jobs_for_target_utilization(
    config: SingleClusterStressConfig,
    target_u: float,
    horizon_slots: int | None = None,
) -> int:
    """Return the job count whose expected utilisation matches ``target_u``.

    Node demand and duration are sampled independently, so
    ``E[d * p] = E[d] * E[p]`` and the expected utilisation of ``n`` jobs is
    ``n * E[d * p] / (C * T)``. Matching ``num_jobs`` to the target keeps the
    accept/reject loop on utilisation from spinning. ``T`` defaults to one day
    of ``config.slot_minutes`` slots.
    """

    if not 0 < target_u <= 1:
        raise ValueError("target_u must be in (0, 1]")
    horizon = DAY_MINUTES // config.slot_minutes if horizon_slots is None else int(horizon_slots)
    expected_nodes = float(np.dot(config.node_demand_values, config.node_demand_probabilities))
    expected_duration = float(np.dot(config.duration_values, config.duration_probabilities))
    return max(1, int(round(target_u * config.node_capacity * horizon / (expected_nodes * expected_duration))))


def expand_hourly_energy_to_slots(hourly_df: pd.DataFrame, slot_minutes: int) -> pd.DataFrame:
    """Repeat hourly power and price inputs over equally sized sub-hour slots.

    Renewable availability is power in MW, so it is repeated rather than
    divided. Energy calculations subsequently multiply by ``delta_t``.
    """

    if slot_minutes <= 0 or 60 % slot_minutes != 0:
        raise ValueError("slot_minutes must be a positive divisor of one hour")
    required = {"hour", "renewable_available", "grid_price", "pue", "baseline_load"}
    missing = required.difference(hourly_df.columns)
    if missing:
        raise ValueError(f"hourly inputs missing required columns: {sorted(missing)}")
    copies = 60 // slot_minutes
    rows: list[dict[str, Any]] = []
    for source in hourly_df.sort_values("hour").itertuples(index=False):
        source_row = source._asdict()
        for offset in range(copies):
            row = dict(source_row)
            row["hour"] = int(source.hour) * copies + offset
            row["source_hour"] = int(source.hour)
            rows.append(row)
    return pd.DataFrame(rows)


def add_generator_effective_price(
    energy_df: pd.DataFrame,
    config: SingleClusterStressConfig,
) -> pd.DataFrame:
    """Estimate a price signal used only to create price-sensitive windows."""

    required = {"grid_price", "renewable_available"}
    missing = required.difference(energy_df.columns)
    if missing:
        raise ValueError(f"energy inputs missing required columns: {sorted(missing)}")
    out = energy_df.sort_values("hour").copy()
    renewable_price = (
        float(out["renewable_price"].iloc[0])
        if "renewable_price" in out.columns
        else 0.8 * float(out["grid_price"].min())
    )
    expected_facility_load = config.target_utilization * config.node_capacity * config.power_per_node_mw * config.pue
    renewable_share = (out["renewable_available"] / expected_facility_load).clip(lower=0.0, upper=1.0)
    out["renewable_price"] = renewable_price
    out["effective_price"] = renewable_share * renewable_price + (1.0 - renewable_share) * out["grid_price"]
    return out


def build_single_a100_cluster(config: SingleClusterStressConfig) -> pd.DataFrame:
    """Build the one-cluster table consumed directly in node scheduling units."""

    raw_gpu_count = config.node_capacity * config.gpus_per_node
    power_capacity_kw = raw_gpu_count * config.gpu_tdp_kw
    return pd.DataFrame([
        {
            "cluster_id": "a100_cluster",
            "cluster_role": "single_a100_node_cluster",
            "gpu_type": "NVIDIA A100-SXM 80GB",
            "node_capacity": config.node_capacity,
            "raw_gpu_count": raw_gpu_count,
            # Generic solver fields use node units in this experiment.
            "gpu_count": config.node_capacity,
            "gpu_capacity": config.node_capacity,
            "capacity_kw": power_capacity_kw,
            "power_capacity_kw": power_capacity_kw,
            "capacity": power_capacity_kw / 1000.0,
            "cpu_capacity": 0.0,
            "memory_capacity_gb": 0.0,
        }
    ])


def _start_costs(effective_price: np.ndarray, duration: int) -> np.ndarray:
    """Return price-only proxy cost for every possible start of one duration."""

    return np.array([
        effective_price[start:start + duration].sum()
        for start in range(len(effective_price) - duration + 1)
    ])


def _window_candidates(
    hidden_start: int,
    duration: int,
    start_costs: np.ndarray,
    config: SingleClusterStressConfig,
    required_active_slot: int | None = None,
) -> list[tuple[int, int, float]]:
    """Return windows containing the hidden start with enough price variation."""

    candidates: list[tuple[int, int, float]] = []
    for options in range(config.min_start_options, config.max_start_options + 1):
        lower = max(0, hidden_start - options + 1)
        upper = min(hidden_start, len(start_costs) - options)
        for earliest in range(lower, upper + 1):
            latest = earliest + options - 1
            costs = start_costs[earliest:latest + 1]
            if costs.min() <= 0:
                continue
            if required_active_slot is not None and not any(
                start <= required_active_slot < start + duration
                for start in range(earliest, latest + 1)
            ):
                continue
            ratio = float(costs.max() / costs.min())
            if ratio >= config.min_price_spread_ratio:
                candidates.append((earliest, latest, ratio))
    return candidates


def _place_hidden_schedule(
    nodes: np.ndarray,
    durations: np.ndarray,
    eligible_starts: dict[int, np.ndarray],
    node_capacity: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Greedily construct one feasible schedule used only by the generator."""

    horizon_slots = max(starts.max() + int(durations[index]) for index, starts in eligible_starts.items())
    load = np.zeros(horizon_slots, dtype=int)
    starts = np.full(len(nodes), -1, dtype=int)
    order = sorted(range(len(nodes)), key=lambda index: int(nodes[index] * durations[index]), reverse=True)
    for index in order:
        choices: list[tuple[int, int, np.ndarray]] = []
        for start in eligible_starts[index]:
            end = int(start + durations[index])
            candidate = load.copy()
            candidate[int(start):end] += int(nodes[index])
            if candidate.max() <= node_capacity:
                choices.append((int(candidate.max()), int(start), candidate))
        if not choices:
            return None
        min_peak = min(choice[0] for choice in choices)
        tied = [choice for choice in choices if choice[0] == min_peak]
        _, start, load = tied[int(rng.integers(len(tied)))]
        starts[index] = start
    return starts, load


def _place_random_feasible_schedule(
    nodes: np.ndarray,
    durations: np.ndarray,
    eligible_starts: dict[int, np.ndarray],
    node_capacity: int,
    rng: np.random.Generator,
    restarts: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Plant a schedule with random job order and a random capacity-feasible start.

    Unlike the greedy min-peak rule, the certificate carries no load-balancing
    structure that a heuristic could exploit. A dead end restarts the whole
    placement, up to ``restarts`` times.
    """

    horizon_slots = max(starts.max() + int(durations[index]) for index, starts in eligible_starts.items())
    for _ in range(restarts):
        load = np.zeros(horizon_slots, dtype=int)
        starts = np.full(len(nodes), -1, dtype=int)
        complete = True
        for index in rng.permutation(len(nodes)):
            duration = int(durations[index])
            demand = int(nodes[index])
            feasible = [
                int(start)
                for start in eligible_starts[int(index)]
                if load[int(start):int(start) + duration].max() + demand <= node_capacity
            ]
            if not feasible:
                complete = False
                break
            start = feasible[int(rng.integers(len(feasible)))]
            load[start:start + duration] += demand
            starts[index] = start
        if complete:
            return starts, load
    return None


def _starts_near_required_slot(
    duration: int,
    horizon_slots: int,
    required_active_slot: int,
    max_start_options: int,
) -> np.ndarray:
    """Return hidden starts that can later expose an option active at one slot."""

    active_starts = range(
        max(0, required_active_slot - duration + 1),
        min(required_active_slot, horizon_slots - duration) + 1,
    )
    allowed: set[int] = set()
    for active_start in active_starts:
        lower = max(0, active_start - max_start_options + 1)
        upper = min(horizon_slots - duration, active_start + max_start_options - 1)
        allowed.update(range(lower, upper + 1))
    return np.array(sorted(allowed), dtype=int)


def potential_active_load(jobs_df: pd.DataFrame, horizon_slots: int) -> np.ndarray:
    """Compute the requested potential active node load for every slot.

    A job contributes to a slot when at least one of its feasible starts would
    make it active in that slot. This is a conflict-pressure diagnostic, not a
    physically realizable schedule load.
    """

    load = np.zeros(horizon_slots, dtype=int)
    for job in jobs_df.itertuples(index=False):
        for slot in range(horizon_slots):
            can_be_active = any(
                start <= slot < start + int(job.duration)
                for start in range(int(job.earliest_start), int(job.latest_start) + 1)
            )
            if can_be_active:
                load[slot] += int(job.node_units_required)
    return load


def generate_single_cluster_stress_instance(
    energy_df: pd.DataFrame,
    config: SingleClusterStressConfig,
    seed: int,
    instance_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Generate a price-sensitive, planted-feasible one-cluster benchmark.

    The returned hidden schedule certifies feasibility but must not be given to
    a solver. The exposed jobs table contains only demand, duration, and a
    flexible start window.
    """

    energy = add_generator_effective_price(energy_df, config)
    horizon_slots = len(energy)
    if max(config.duration_values) > horizon_slots:
        raise ValueError("the longest configured duration exceeds the horizon")
    if config.max_start_options > horizon_slots - min(config.duration_values) + 1:
        raise ValueError("max_start_options exceeds the feasible start range")

    effective_price = energy["effective_price"].to_numpy(dtype=float)
    cheap_threshold = float(np.quantile(effective_price, config.cheap_price_quantile))
    cheap_slots = energy.loc[energy["effective_price"] <= cheap_threshold, "hour"].astype(int).to_numpy()
    cheapest_slot = int(energy.loc[energy["effective_price"].idxmin(), "hour"])
    rng = np.random.default_rng(seed)
    total_capacity_slots = config.node_capacity * horizon_slots
    rejections = {"utilization": 0, "eligible_starts": 0, "planting": 0, "window": 0, "cheap_peak": 0}
    for attempt in range(1, config.max_attempts + 1):
        nodes = rng.choice(config.node_demand_values, size=config.num_jobs, p=config.node_demand_probabilities).astype(int)
        durations = rng.choice(config.duration_values, size=config.num_jobs, p=config.duration_probabilities).astype(int)
        utilization = float((nodes * durations).sum() / total_capacity_slots)
        if abs(utilization - config.target_utilization) > config.utilization_tolerance:
            rejections["utilization"] += 1
            continue

        start_costs = {duration: _start_costs(effective_price, int(duration)) for duration in set(durations)}
        # Select enough large jobs to make the cheapest slot oversubscribed in
        # potential load. Their hidden placements remain feasible elsewhere.
        target_nodes = config.min_peak_potential_load_ratio * config.node_capacity
        selected_for_cheap_slot: set[int] = set()
        accumulated_nodes = 0
        if config.steer_to_cheap_slot:
            for index in sorted(range(config.num_jobs), key=lambda item: int(nodes[item]), reverse=True):
                selected_for_cheap_slot.add(index)
                accumulated_nodes += int(nodes[index])
                if accumulated_nodes >= target_nodes:
                    break
        eligible_starts: dict[int, np.ndarray] = {}
        for index, duration in enumerate(durations):
            duration_costs = start_costs[int(duration)]
            if index in selected_for_cheap_slot:
                nearby = _starts_near_required_slot(
                    int(duration), horizon_slots, cheapest_slot, config.max_start_options
                )
                eligible_starts[index] = np.array(
                    [
                        start
                        for start in nearby
                        if _window_candidates(
                            int(start),
                            int(duration),
                            duration_costs,
                            config,
                            required_active_slot=cheapest_slot,
                        )
                    ],
                    dtype=int,
                )
            else:
                eligible_starts[index] = np.array(
                    [
                        start
                        for start in range(len(duration_costs))
                        if _window_candidates(int(start), int(duration), duration_costs, config)
                    ],
                    dtype=int,
                )
            if len(eligible_starts[index]) == 0:
                break
        if len(eligible_starts) != config.num_jobs or any(len(starts) == 0 for starts in eligible_starts.values()):
            rejections["eligible_starts"] += 1
            continue
        if config.planting == "random_feasible":
            placed = _place_random_feasible_schedule(
                nodes, durations, eligible_starts, config.node_capacity, rng, config.random_planting_restarts
            )
        else:
            placed = _place_hidden_schedule(nodes, durations, eligible_starts, config.node_capacity, rng)
        if placed is None:
            rejections["planting"] += 1
            continue
        hidden_starts, hidden_load = placed

        job_rows: list[dict[str, Any]] = []
        valid = True
        for index, (node_count, duration, hidden_start) in enumerate(zip(nodes, durations, hidden_starts, strict=True)):
            candidates = _window_candidates(
                int(hidden_start),
                int(duration),
                start_costs[int(duration)],
                config,
                required_active_slot=cheapest_slot if index in selected_for_cheap_slot else None,
            )
            if not candidates:
                valid = False
                break
            earliest, latest, price_ratio = candidates[int(rng.integers(len(candidates)))]
            job_rows.append(
                {
                    "job_id": f"{instance_name}_{index:04d}",
                    "category": "synthetic_energy_tradeoff",
                    "workload_family": "single_cluster_stress",
                    "compatible_clusters": "a100_cluster",
                    "received_hour": earliest,
                    "earliest_start": earliest,
                    "latest_start": latest,
                    "duration": int(duration),
                    "node_units_required": int(node_count),
                    "gpu_count_required": int(node_count),
                    "power": float(node_count * config.power_per_node_mw),
                    "power_kw": float(node_count * config.power_per_node_mw * 1000.0),
                    "cpu_required": 0.0,
                    "memory_required_gb": 0.0,
                    "price_spread_ratio": price_ratio,
                }
            )
        if not valid:
            rejections["window"] += 1
            continue
        jobs = pd.DataFrame(job_rows)
        potential_load = potential_active_load(jobs, horizon_slots)
        cheap_peak_ratio = float(potential_load[cheap_slots].max() / config.node_capacity)
        if cheap_peak_ratio < config.min_peak_potential_load_ratio:
            rejections["cheap_peak"] += 1
            continue
        power_intensity = np.ones(config.num_jobs)
        if tuple(config.power_intensity_range) != (1.0, 1.0):
            # Drawn only after acceptance and only for non-default ranges, so
            # the default RNG stream (and output) is unchanged. Node demand is
            # not scaled: only the energy weight of each job changes.
            low, high = config.power_intensity_range
            power_intensity = rng.uniform(low, high, size=config.num_jobs)
            jobs["power"] = jobs["power"] * power_intensity
            jobs["power_kw"] = jobs["power_kw"] * power_intensity

        clusters = build_single_a100_cluster(config)
        hidden = jobs[["job_id", "node_units_required", "duration"]].copy()
        hidden["hidden_start"] = hidden_starts
        diagnostics = pd.DataFrame(
            {
                "hour": energy["hour"].astype(int),
                "effective_price": effective_price,
                "cheap_slot": energy["hour"].isin(cheap_slots).to_numpy(),
                "potential_active_nodes": potential_load,
                "potential_active_load_ratio": potential_load / config.node_capacity,
                "hidden_feasible_nodes": hidden_load,
                "hidden_feasible_load_ratio": hidden_load / config.node_capacity,
            }
        )
        metadata = {
            "generator": "single_cluster_price_sensitive_stress",
            "instance_name": instance_name,
            "seed": seed,
            "attempt": attempt,
            "config": asdict(config),
            "horizon_slots": horizon_slots,
            "total_node_slot_utilization": utilization,
            "mean_duration_slots": float(jobs["duration"].mean()),
            "mean_node_demand": float(jobs["node_units_required"].mean()),
            "multi_node_job_share": float((jobs["node_units_required"] > 1).mean()),
            "mean_start_options": float((jobs["latest_start"] - jobs["earliest_start"] + 1).mean()),
            "mean_price_spread_ratio": float(jobs["price_spread_ratio"].mean()),
            "cheap_slot_count": int(len(cheap_slots)),
            "targeted_cheap_slot": cheapest_slot,
            "targeted_cheap_potential_nodes": int(accumulated_nodes),
            "cheap_peak_potential_load_ratio": cheap_peak_ratio,
            "hidden_peak_load_ratio": float(hidden_load.max() / config.node_capacity),
            "rejections": dict(rejections),
            "power_intensity": [float(value) for value in power_intensity],
        }
        return jobs, clusters, energy, hidden, diagnostics, metadata
    raise RuntimeError(
        "Could not construct an instance meeting the configured stress criteria "
        f"after {config.max_attempts} attempts; rejections={rejections}"
    )
