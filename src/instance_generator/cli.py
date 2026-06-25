"""Command-line entry point for synthetic instance generation."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from src.instance_generator.generator import SAMPLING_MODES, SyntheticInstanceConfig, generate_feasible_instance


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a feasible synthetic MILP stress-test instance.")
    parser.add_argument("--num-jobs", type=int, required=True, help="Number of jobs to generate.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory where generated files are written.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--horizon-hours", type=int, default=24)
    parser.add_argument("--slot-minutes", type=int, default=60)
    parser.add_argument("--sampling-mode", choices=sorted(SAMPLING_MODES), default="alibaba")
    parser.add_argument(
        "--alibaba-calibration-dir",
        type=Path,
        default=Path("experiments/outputs/alibaba_2023_eda"),
    )
    parser.add_argument("--min-duration", type=int, default=1)
    parser.add_argument("--max-duration", type=int, default=6)
    parser.add_argument("--min-gpu-demand", type=int, default=1)
    parser.add_argument("--max-gpu-demand", type=int, default=4)
    parser.add_argument("--power-per-gpu-kw", type=float, default=0.35)
    parser.add_argument("--min-window-slack", type=int, default=2)
    parser.add_argument("--max-window-slack", type=int, default=8)
    parser.add_argument("--extra-compatibility-probability", type=float, default=0.75)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = SyntheticInstanceConfig(
        num_jobs=args.num_jobs,
        horizon_hours=args.horizon_hours,
        slot_minutes=args.slot_minutes,
        seed=args.seed,
        sampling_mode=args.sampling_mode,
        alibaba_calibration_dir=args.alibaba_calibration_dir,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
        min_gpu_demand=args.min_gpu_demand,
        max_gpu_demand=args.max_gpu_demand,
        power_per_gpu_kw=args.power_per_gpu_kw,
        min_window_slack=args.min_window_slack,
        max_window_slack=args.max_window_slack,
        extra_compatibility_probability=args.extra_compatibility_probability,
    )
    instance = generate_feasible_instance(config)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    instance.jobs_df.to_csv(args.output_dir / "jobs.csv", index=False)
    instance.hourly_df.to_csv(args.output_dir / "hourly_inputs.csv", index=False)
    instance.clusters_df.to_csv(args.output_dir / "clusters.csv", index=False)
    instance.hidden_schedule_df.to_csv(args.output_dir / "hidden_schedule.csv", index=False)
    (args.output_dir / "config.json").write_text(json.dumps(asdict(instance.config), indent=2), encoding="utf-8")
    (args.output_dir / "feasibility_report.json").write_text(
        json.dumps(asdict(instance.feasibility_report), indent=2),
        encoding="utf-8",
    )
    print(f"Wrote feasible synthetic instance with {len(instance.jobs_df)} jobs to {args.output_dir}")


if __name__ == "__main__":
    main()
