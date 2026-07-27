"""Create reusable hard benchmark instances for classical and quantum tests."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.instance_generator.hard_instances import create_hard_benchmark_pack


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "experiments" / "combinatorial_difficulty" / "constrained_benchmark_instances",
        help="Directory where instance folders and manifest.csv will be written.",
    )
    return parser.parse_args()


def main() -> None:
    """Create the curated hard-instance benchmark pack."""

    args = parse_args()
    manifest = create_hard_benchmark_pack(args.output_dir)
    print(f"Wrote {len(manifest)} hard benchmark instances to {args.output_dir}")
    print(args.output_dir / "manifest.csv")


if __name__ == "__main__":
    main()
