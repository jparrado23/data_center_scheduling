"""Screen one-cluster 15-minute stress profiles with Gurobi.

The profiles target the mechanisms that made notebook 10 difficult while
preserving the current single-cluster formulation: multi-node packing,
multi-slot duration, price-sensitive windows, high utilization, and optional
contracted-power pressure.
"""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import time

import pandas as pd

from src.config import ModelConfig
from src.data.energy_prices import load_marginalpdbc
from src.data.scenarios import ENERGY_SCENARIOS
from src.data.solar_profiles import build_daily_hourly_solar_profile
from src.instance_generator.single_cluster_stress import (
    SingleClusterStressConfig,
    expand_hourly_energy_to_slots,
    generate_single_cluster_stress_instance,
)
from src.milp.gurobi_model import build_milp_model
from src.milp.solve import solve_model


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "data" / "instances" / "single_cluster_15m" / "stress_sweep"
SOLAR_PATH = PROJECT_ROOT / "data" / "energy" / "solar" / "Timeseries_40.409_-3.698_SA3_1000kWp_crystSi_14_36deg_-5deg_2020_2023.csv"
SCENARIO_NAME = "base"
SEEDS = (7401, 7402, 7403)
TIME_LIMIT_SECONDS = 60
MIP_GAP = 0.001


PROFILE_SPECS = [
    {
        "name": "reference_many_small",
        "contracted_power_ratio": 1.00,
        "config": SingleClusterStressConfig(
            num_jobs=180,
            target_utilization=0.80,
            node_demand_values=(8, 16, 32, 64),
            node_demand_probabilities=(0.45, 0.30, 0.18, 0.07),
            duration_values=(2, 4, 8, 12),
            duration_probabilities=(0.25, 0.35, 0.25, 0.15),
            min_start_options=6,
            max_start_options=12,
        ),
    },
    {
        "name": "packing_medium_large",
        "contracted_power_ratio": 0.70,
        "config": SingleClusterStressConfig(
            num_jobs=72,
            target_utilization=0.80,
            utilization_tolerance=0.08,
            node_demand_values=(16, 32, 64, 96),
            node_demand_probabilities=(0.20, 0.35, 0.30, 0.15),
            duration_values=(2, 4, 8, 12),
            duration_probabilities=(0.25, 0.35, 0.25, 0.15),
            min_start_options=6,
            max_start_options=12,
        ),
    },
    {
        "name": "chunky_fragmentation",
        "contracted_power_ratio": 0.70,
        "config": SingleClusterStressConfig(
            num_jobs=55,
            target_utilization=0.80,
            utilization_tolerance=0.08,
            node_demand_values=(32, 64, 96, 128),
            node_demand_probabilities=(0.15, 0.40, 0.30, 0.15),
            duration_values=(2, 4, 8),
            duration_probabilities=(0.20, 0.45, 0.35),
            min_start_options=6,
            max_start_options=12,
        ),
    },
    {
        "name": "long_duration_tradeoff",
        "contracted_power_ratio": 0.70,
        "config": SingleClusterStressConfig(
            num_jobs=42,
            target_utilization=0.82,
            utilization_tolerance=0.08,
            node_demand_values=(16, 32, 64, 96),
            node_demand_probabilities=(0.15, 0.35, 0.35, 0.15),
            duration_values=(4, 8, 12, 16),
            duration_probabilities=(0.15, 0.35, 0.35, 0.15),
            min_start_options=8,
            max_start_options=16,
        ),
    },
    {
        "name": "broad_choice_many_small",
        "contracted_power_ratio": 0.70,
        "config": SingleClusterStressConfig(
            num_jobs=180,
            target_utilization=0.80,
            node_demand_values=(8, 16, 32, 64),
            node_demand_probabilities=(0.45, 0.30, 0.18, 0.07),
            duration_values=(2, 4, 8, 12),
            duration_probabilities=(0.25, 0.35, 0.25, 0.15),
            min_start_options=12,
            max_start_options=24,
        ),
    },
]


def build_energy() -> pd.DataFrame:
    """Load the configured real daily profile and expand it to 15-minute slots."""

    scenario = ENERGY_SCENARIOS[SCENARIO_NAME]
    prices = load_marginalpdbc(PROJECT_ROOT / scenario["price_file"], price_column="last")
    solar = build_daily_hourly_solar_profile(SOLAR_PATH, scenario["date"])
    hourly = prices[["hour", "grid_price"]].merge(
        solar[["hour", "renewable_available"]], on="hour", validate="one_to_one"
    )
    hourly["pue"] = 1.25
    hourly["baseline_load"] = 0.0
    hourly["renewable_price"] = 0.8 * float(hourly["grid_price"].min())
    return expand_hourly_energy_to_slots(hourly, 15)


def solve_profile(
    name: str,
    config: SingleClusterStressConfig,
    contracted_power_ratio: float,
    seed: int,
    energy_df: pd.DataFrame,
) -> dict[str, object]:
    """Generate, persist, and solve one profile/seed combination."""

    instance_name = f"{name}_seed_{seed}"
    jobs, clusters, energy, hidden, diagnostics, metadata = generate_single_cluster_stress_instance(
        energy_df, config, seed, instance_name
    )
    instance_dir = OUTPUT_ROOT / instance_name
    instance_dir.mkdir(parents=True, exist_ok=True)
    jobs.to_csv(instance_dir / "jobs.csv", index=False)
    clusters.to_csv(instance_dir / "clusters.csv", index=False)
    energy.to_csv(instance_dir / "energy_inputs_15m.csv", index=False)
    hidden.to_csv(instance_dir / "hidden_feasibility_certificate.csv", index=False)
    diagnostics.to_csv(instance_dir / "generation_diagnostics.csv", index=False)
    (instance_dir / "generation_metadata.json").write_text(json.dumps(metadata, indent=2))

    model_config = ModelConfig(
        contracted_power=float(clusters["capacity"].sum()) * config.pue * contracted_power_ratio,
        renewable_price=float(energy["renewable_price"].iloc[0]),
        peak_price=1000.0,
        delta_t=config.delta_t,
        pue=config.pue,
    )
    model, variables = build_milp_model(
        jobs,
        energy[["hour", "renewable_available", "grid_price", "pue", "baseline_load"]],
        clusters,
        model_config,
        model_name=instance_name,
        enforce_gpu_constraints=True,
        enforce_cpu_constraints=False,
        enforce_memory_constraints=False,
    )
    model.Params.OutputFlag = 0
    started = time.perf_counter()
    solve_model(model, time_limit=TIME_LIMIT_SECONDS, mip_gap=MIP_GAP)
    wall_seconds = time.perf_counter() - started
    row = {
        "instance": instance_name,
        "profile": name,
        "seed": seed,
        "status": int(model.Status),
        "has_solution": bool(model.SolCount),
        "runtime_s": wall_seconds,
        "mip_gap": float(model.MIPGap) if model.SolCount else None,
        "objective": float(model.ObjVal) if model.SolCount else None,
        "best_bound": float(model.ObjBound) if model.SolCount else None,
        "assignment_variables": len(variables["x"]),
        "constraints": int(model.NumConstrs),
        "contracted_power_ratio": contracted_power_ratio,
        "utilization": metadata["total_node_slot_utilization"],
        "mean_duration_slots": metadata["mean_duration_slots"],
        "mean_node_demand": metadata["mean_node_demand"],
        "mean_start_options": metadata["mean_start_options"],
        "mean_price_spread_ratio": metadata["mean_price_spread_ratio"],
        "cheap_peak_potential_load_ratio": metadata["cheap_peak_potential_load_ratio"],
        "hidden_peak_load_ratio": metadata["hidden_peak_load_ratio"],
        "config": json.dumps(asdict(config)),
    }
    pd.DataFrame([row]).to_csv(instance_dir / "gurobi_summary.csv", index=False)
    return row


def main() -> None:
    """Execute the screening sweep and persist one compact result table."""

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    energy = build_energy()
    rows: list[dict[str, object]] = []
    for spec in PROFILE_SPECS:
        for seed in SEEDS:
            instance_name = f"{spec['name']}_seed_{seed}"
            existing_summary = OUTPUT_ROOT / instance_name / "gurobi_summary.csv"
            if existing_summary.exists():
                rows.append(pd.read_csv(existing_summary).iloc[0].to_dict())
                continue
            print(f"Solving {spec['name']} / seed {seed}", flush=True)
            try:
                rows.append(
                    solve_profile(spec["name"], spec["config"], spec["contracted_power_ratio"], seed, energy)
                )
            except RuntimeError as error:
                rows.append({"profile": spec["name"], "seed": seed, "generation_error": str(error)})
    summary = pd.DataFrame(rows)
    summary.to_csv(OUTPUT_ROOT / "gurobi_stress_sweep_summary.csv", index=False)
    print(summary.sort_values("runtime_s", ascending=False, na_position="last").to_string(index=False))


if __name__ == "__main__":
    main()
