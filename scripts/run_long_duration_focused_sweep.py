"""Focused search for optimization-hard one-cluster 15-minute instances."""

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
OUTPUT_ROOT = PROJECT_ROOT / "data" / "instances" / "single_cluster_15m" / "long_duration_focused"
SOLAR_PATH = PROJECT_ROOT / "data" / "energy" / "solar" / "Timeseries_40.409_-3.698_SA3_1000kWp_crystSi_14_36deg_-5deg_2020_2023.csv"
SCENARIO_NAME = "base"
SEEDS = (7601, 7602, 7603)
TIME_LIMIT_SECONDS = 60
MIP_GAP = 0.001


def long_config(
    *,
    num_jobs: int,
    target_utilization: float,
    start_options: tuple[int, int],
) -> SingleClusterStressConfig:
    """Return one long-duration, medium/large-node workload profile."""

    return SingleClusterStressConfig(
        num_jobs=num_jobs,
        target_utilization=target_utilization,
        utilization_tolerance=0.06,
        node_demand_values=(16, 32, 64, 96),
        node_demand_probabilities=(0.15, 0.35, 0.35, 0.15),
        duration_values=(4, 8, 12, 16),
        duration_probabilities=(0.15, 0.35, 0.35, 0.15),
        min_start_options=start_options[0],
        max_start_options=start_options[1],
        min_price_spread_ratio=1.15,
        min_peak_potential_load_ratio=1.50,
        # A focused experiment should record a rejected profile promptly
        # instead of spending minutes searching for a rare construction.
        max_attempts=100,
    )


PROFILE_SPECS = [
    {"name": "long_base", "contract_ratio": 0.70, "config": long_config(num_jobs=42, target_utilization=0.82, start_options=(8, 16))},
    {"name": "long_wide_window", "contract_ratio": 0.70, "config": long_config(num_jobs=42, target_utilization=0.82, start_options=(12, 20))},
    {"name": "long_high_util", "contract_ratio": 0.70, "config": long_config(num_jobs=45, target_utilization=0.90, start_options=(8, 16))},
    {"name": "long_high_util_wide", "contract_ratio": 0.70, "config": long_config(num_jobs=45, target_utilization=0.90, start_options=(12, 20))},
    {"name": "long_contract_60", "contract_ratio": 0.60, "config": long_config(num_jobs=42, target_utilization=0.82, start_options=(8, 16))},
    {"name": "long_high_util_contract_60", "contract_ratio": 0.60, "config": long_config(num_jobs=45, target_utilization=0.90, start_options=(12, 20))},
]


def build_energy() -> pd.DataFrame:
    """Build the base daily profile at 15-minute resolution."""

    scenario = ENERGY_SCENARIOS[SCENARIO_NAME]
    price = load_marginalpdbc(PROJECT_ROOT / scenario["price_file"], price_column="last")
    solar = build_daily_hourly_solar_profile(SOLAR_PATH, scenario["date"])
    hourly = price[["hour", "grid_price"]].merge(
        solar[["hour", "renewable_available"]], on="hour", validate="one_to_one"
    )
    hourly["pue"] = 1.25
    hourly["baseline_load"] = 0.0
    hourly["renewable_price"] = 0.8 * float(hourly["grid_price"].min())
    return expand_hourly_energy_to_slots(hourly, 15)


def solve_one(spec: dict[str, object], seed: int, energy: pd.DataFrame) -> dict[str, object]:
    """Generate, persist, and solve one focused candidate."""

    config = spec["config"]
    assert isinstance(config, SingleClusterStressConfig)
    contract_ratio = float(spec["contract_ratio"])
    name = f"{spec['name']}_seed_{seed}"
    output_dir = OUTPUT_ROOT / name
    existing = output_dir / "gurobi_summary.csv"
    if existing.exists():
        return pd.read_csv(existing).iloc[0].to_dict()

    jobs, clusters, energy_inputs, hidden, diagnostics, metadata = generate_single_cluster_stress_instance(
        energy, config, seed, name
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    jobs.to_csv(output_dir / "jobs.csv", index=False)
    clusters.to_csv(output_dir / "clusters.csv", index=False)
    energy_inputs.to_csv(output_dir / "energy_inputs_15m.csv", index=False)
    hidden.to_csv(output_dir / "hidden_feasibility_certificate.csv", index=False)
    diagnostics.to_csv(output_dir / "generation_diagnostics.csv", index=False)
    (output_dir / "generation_metadata.json").write_text(json.dumps(metadata, indent=2))

    model_config = ModelConfig(
        contracted_power=float(clusters["capacity"].sum()) * config.pue * contract_ratio,
        renewable_price=float(energy_inputs["renewable_price"].iloc[0]),
        peak_price=1000.0,
        delta_t=config.delta_t,
        pue=config.pue,
    )
    model, variables = build_milp_model(
        jobs,
        energy_inputs[["hour", "renewable_available", "grid_price", "pue", "baseline_load"]],
        clusters,
        model_config,
        model_name=name,
        enforce_gpu_constraints=True,
        enforce_cpu_constraints=False,
        enforce_memory_constraints=False,
    )
    model.Params.OutputFlag = 0
    started = time.perf_counter()
    solve_model(model, time_limit=TIME_LIMIT_SECONDS, mip_gap=MIP_GAP)
    row = {
        "instance": name,
        "profile": spec["name"],
        "seed": seed,
        "status": int(model.Status),
        "has_solution": bool(model.SolCount),
        "runtime_s": time.perf_counter() - started,
        "mip_gap": float(model.MIPGap) if model.SolCount else None,
        "objective": float(model.ObjVal) if model.SolCount else None,
        "best_bound": float(model.ObjBound) if model.SolCount else None,
        "assignment_variables": len(variables["x"]),
        "constraints": int(model.NumConstrs),
        "contracted_power_ratio": contract_ratio,
        "utilization": metadata["total_node_slot_utilization"],
        "mean_duration_slots": metadata["mean_duration_slots"],
        "mean_node_demand": metadata["mean_node_demand"],
        "mean_start_options": metadata["mean_start_options"],
        "mean_price_spread_ratio": metadata["mean_price_spread_ratio"],
        "cheap_peak_potential_load_ratio": metadata["cheap_peak_potential_load_ratio"],
        "hidden_peak_load_ratio": metadata["hidden_peak_load_ratio"],
        "config": json.dumps(asdict(config)),
    }
    pd.DataFrame([row]).to_csv(existing, index=False)
    return row


def main() -> None:
    """Run the focused sweep and save an aggregated result table."""

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    energy = build_energy()
    rows: list[dict[str, object]] = []
    for spec in PROFILE_SPECS:
        for seed in SEEDS:
            print(f"Solving {spec['name']} / seed {seed}", flush=True)
            try:
                rows.append(solve_one(spec, seed, energy))
            except RuntimeError as error:
                rows.append({"profile": spec["name"], "seed": seed, "generation_error": str(error)})
            pd.DataFrame(rows).to_csv(OUTPUT_ROOT / "focused_sweep_partial.csv", index=False)
    summary = pd.DataFrame(rows)
    summary.to_csv(OUTPUT_ROOT / "focused_sweep_summary.csv", index=False)
    print(summary.sort_values("runtime_s", ascending=False, na_position="last").to_string(index=False))


if __name__ == "__main__":
    main()
