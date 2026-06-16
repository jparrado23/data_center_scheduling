"""Solve a repo-local thesis scenario with Gurobi."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.scenarios import ENERGY_SCENARIOS, WORKLOAD_FILES, build_repo_scenario
from src.evaluation.metrics import compute_summary_metrics
from src.evaluation.results import extract_cluster_hourly_results, extract_hourly_results, extract_schedule
from src.milp.gurobi_model import build_milp_model
from src.milp.solve import solve_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", choices=sorted(WORKLOAD_FILES), default="tense")
    parser.add_argument(
        "--jobs-csv",
        type=Path,
        help="Specific job instance CSV. Relative paths are resolved from the repo root.",
    )
    parser.add_argument("--scenario", choices=sorted(ENERGY_SCENARIOS), default="base")
    parser.add_argument("--solar-profile", type=Path, default=Path("data/solar_profile/monthly_solar_profiles.csv"))
    parser.add_argument("--baseline-load-mw", type=float, default=0.010)
    parser.add_argument("--renewable-price", type=float, default=40.0)
    parser.add_argument("--peak-price", type=float, default=1000.0)
    parser.add_argument("--contracted-power", type=float, default=0.222)
    parser.add_argument("--price-column", choices=["first", "last", "average"], default="last")
    parser.add_argument("--time-limit", type=float, default=60.0)
    parser.add_argument("--mip-gap", type=float)
    parser.add_argument("--quiet", action="store_true", help="Disable Gurobi solver log output.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory. Defaults to experiments/outputs/<workload>_<scenario>.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or PROJECT_ROOT / "experiments" / "outputs" / f"{args.workload}_{args.scenario}"

    jobs_df, hourly_df, clusters_df, config = build_repo_scenario(
        project_root=PROJECT_ROOT,
        workload_case=args.workload,
        jobs_csv=args.jobs_csv,
        energy_scenario=args.scenario,
        solar_profile_csv=args.solar_profile,
        baseline_load_mw=args.baseline_load_mw,
        renewable_price=args.renewable_price,
        peak_price=args.peak_price,
        contracted_power=args.contracted_power,
        price_column=args.price_column,
    )

    model, variables = build_milp_model(jobs_df, hourly_df, clusters_df, config)
    model.Params.OutputFlag = 0 if args.quiet else 1
    solve_model(model, time_limit=args.time_limit, mip_gap=args.mip_gap)

    schedule_df = extract_schedule(jobs_df, variables)
    hourly_results = extract_hourly_results(hourly_df, variables)
    cluster_hourly_results = extract_cluster_hourly_results(variables)
    metrics = compute_summary_metrics(hourly_results, config)

    output_dir.mkdir(parents=True, exist_ok=True)
    schedule_df.to_csv(output_dir / "schedule.csv", index=False)
    hourly_results.to_csv(output_dir / "hourly_results.csv", index=False)
    cluster_hourly_results.to_csv(output_dir / "cluster_hourly_results.csv", index=False)
    pd.Series(metrics).to_csv(output_dir / "metrics.csv", header=["value"])

    print(f"Status: {model.Status}")
    print(f"Solutions: {model.SolCount}")
    print(f"Objective: {model.ObjVal:,.6f}")
    print(f"Decision variables: {len(variables['x'])}")
    for key in [
        "total_cost",
        "energy_cost",
        "grid_cost",
        "renewable_cost",
        "peak_load",
        "peak_over_contracted",
        "peak_cost",
        "renewable_available_mwh",
        "renewable_energy_mwh",
        "renewable_curtailment_mwh",
        "grid_energy_mwh",
    ]:
        print(f"{key}: {metrics[key]:,.6f}")
    print(f"Saved outputs to {output_dir}")


if __name__ == "__main__":
    main()
