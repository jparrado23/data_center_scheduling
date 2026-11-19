"""Tests for daily energy profiles and price-shape perturbations."""

import numpy as np
import pandas as pd
import pytest

from src.instance_generator.price_shapes import apply_price_jitter, apply_two_valleys, build_energy_for_day


@pytest.fixture(scope="module")
def base_energy() -> pd.DataFrame:
    return build_energy_for_day("base")


@pytest.mark.parametrize("scenario", ["base", "clear_sky", "overcast"])
def test_build_energy_for_day_has_96_positive_slots(scenario: str) -> None:
    energy = build_energy_for_day(scenario)

    assert len(energy) == 96
    assert energy["hour"].tolist() == list(range(96))
    assert (energy["grid_price"] > 0).all()


def test_two_valleys_lowers_only_night_slots(base_energy: pd.DataFrame) -> None:
    night = range(12, 24)
    shaped = apply_two_valleys(base_energy, night, depth=0.1)
    level = np.quantile(base_energy["grid_price"], 0.1)

    assert len(shaped) == 96
    assert (shaped["grid_price"] > 0).all()
    assert (shaped.loc[12:23, "grid_price"] <= level + 1e-12).all()
    pd.testing.assert_series_equal(shaped.loc[24:, "grid_price"], base_energy.loc[24:, "grid_price"])


def test_jitter_is_reproducible_and_positive(base_energy: pd.DataFrame) -> None:
    first = apply_price_jitter(base_energy, 0.01, seed=3)
    second = apply_price_jitter(base_energy, 0.01, seed=3)
    other = apply_price_jitter(base_energy, 0.01, seed=4)

    assert len(first) == 96
    assert (first["grid_price"] > 0).all()
    pd.testing.assert_frame_equal(first, second)
    assert not np.allclose(first["grid_price"], other["grid_price"])
    assert not np.allclose(first["grid_price"], base_energy["grid_price"])


def test_jitter_clips_to_positive(base_energy: pd.DataFrame) -> None:
    assert (apply_price_jitter(base_energy, 5.0, seed=0)["grid_price"] > 0).all()
