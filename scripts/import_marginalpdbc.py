"""Convert an OMIE MARGINALPDBC price file into model-ready hourly inputs."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.energy_prices import write_hourly_inputs_from_marginalpdbc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("price_file", type=Path, help="Path to a MARGINALPDBC text file.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/hourly_inputs.csv"),
        help="Path for the normalized hourly_inputs.csv output.",
    )
    parser.add_argument(
        "--existing-hourly-inputs",
        type=Path,
        help="Optional existing hourly_inputs.csv. If provided, only grid_price is replaced.",
    )
    parser.add_argument(
        "--price-column",
        choices=["first", "last", "average"],
        default="last",
        help="Which of the two source price columns to use.",
    )
    parser.add_argument(
        "--renewable-available",
        type=float,
        default=0.0,
        help="Renewable availability to use when creating a new hourly_inputs.csv from prices only.",
    )
    parser.add_argument(
        "--zero-based-source-hours",
        action="store_true",
        help="Use this only if the source hour column is already 0-based.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    hourly_df = write_hourly_inputs_from_marginalpdbc(
        args.price_file,
        args.output,
        existing_hourly_inputs_path=args.existing_hourly_inputs,
        price_column=args.price_column,
        renewable_available=args.renewable_available,
        source_hours_are_one_based=not args.zero_based_source_hours,
    )
    print(f"Wrote {len(hourly_df)} hourly rows to {args.output}")


if __name__ == "__main__":
    main()
