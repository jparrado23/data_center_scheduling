"""Daily 15-minute energy profiles and price-shape perturbations.

These helpers prepare the energy table *before* instance generation, so the
exported ``energy_inputs_15m.csv`` contains the modified prices and the
generator's ``effective_price`` is derived from them.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.energy_prices import load_marginalpdbc
from src.data.scenarios import ENERGY_SCENARIOS
from src.data.solar_profiles import build_daily_hourly_solar_profile
from src.instance_generator.single_cluster_stress import expand_hourly_energy_to_slots


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOLAR_PATH = (
    PROJECT_ROOT
    / "data"
    / "energy"
    / "solar"
    / "Timeseries_40.409_-3.698_SA3_1000kWp_crystSi_14_36deg_-5deg_2020_2023.csv"
)
MIN_PRICE = 1e-6


def build_energy_for_day(
    scenario_name: str,
    *,
    slot_minutes: int = 15,
    pue: float = 1.25,
    project_root: str | Path = PROJECT_ROOT,
    solar_path: str | Path = DEFAULT_SOLAR_PATH,
) -> pd.DataFrame:
    """Build one representative day at sub-hour resolution.

    Generalises ``build_energy`` from ``scripts/run_long_duration_focused_sweep.py``
    to any key of ``ENERGY_SCENARIOS``. With the defaults and ``"base"`` it
    returns exactly the profile used for the ``long_duration_focused`` pack.
    """

    if scenario_name not in ENERGY_SCENARIOS:
        raise KeyError(f"unknown energy scenario {scenario_name!r}; choose from {sorted(ENERGY_SCENARIOS)}")
    scenario = ENERGY_SCENARIOS[scenario_name]
    price = load_marginalpdbc(Path(project_root) / scenario["price_file"], price_column="last")
    solar = build_daily_hourly_solar_profile(solar_path, scenario["date"])
    hourly = price[["hour", "grid_price"]].merge(
        solar[["hour", "renewable_available"]], on="hour", validate="one_to_one"
    )
    hourly["pue"] = pue
    hourly["baseline_load"] = 0.0
    hourly["renewable_price"] = 0.8 * float(hourly["grid_price"].min())
    return expand_hourly_energy_to_slots(hourly, slot_minutes)


def apply_two_valleys(energy: pd.DataFrame, night_slots: Iterable[int], depth: float) -> pd.DataFrame:
    """Add a second cheap valley by lowering night prices to a price quantile.

    ``depth`` is the quantile of the original ``grid_price`` (0 = daily
    minimum). Night slots priced above that quantile are lowered to it; cheaper
    slots are left unchanged. A second valley splits the attractive capacity
    across two separated periods, so jobs compete for two cheap regions.
    """

    if not 0 <= depth <= 1:
        raise ValueError("depth must be a quantile in [0, 1]")
    out = energy.copy()
    level = float(np.quantile(out["grid_price"].to_numpy(dtype=float), depth))
    mask = out["hour"].astype(int).isin([int(slot) for slot in night_slots])
    out.loc[mask, "grid_price"] = np.minimum(out.loc[mask, "grid_price"].astype(float), level)
    return out


def apply_price_jitter(energy: pd.DataFrame, rel_std: float, seed: int) -> pd.DataFrame:
    """Multiply ``grid_price`` by ``1 + eps`` with ``eps ~ N(0, rel_std)``.

    Hourly prices repeated over four 15-minute slots create many exactly tied
    start costs. Small multiplicative noise breaks those ties. Prices are
    clipped to stay strictly positive. The result is reproducible per seed.
    """

    if rel_std < 0:
        raise ValueError("rel_std must be non-negative")
    out = energy.copy()
    rng = np.random.default_rng(seed)
    epsilon = rng.normal(0.0, rel_std, size=len(out))
    out["grid_price"] = np.clip(out["grid_price"].to_numpy(dtype=float) * (1.0 + epsilon), MIN_PRICE, None)
    return out
