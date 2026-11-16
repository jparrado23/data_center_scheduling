from pathlib import Path

import pandas as pd

from src.data.energy_prices import (
    build_hourly_inputs_from_marginalpdbc,
    load_marginalpdbc,
    update_hourly_inputs_prices,
    write_hourly_inputs_from_marginalpdbc,
)
from src.data.validation import validate_hourly_inputs


def _write_price_file(path: Path) -> None:
    rows = ["MARGINALPDBC;"]
    for hour in range(1, 25):
        price_1 = 100 + hour
        price_2 = 200 + hour
        rows.append(f"2023;04;04;{hour};{price_1};{price_2};")
    rows.append("*")
    path.write_text("\n".join(rows), encoding="utf-8")


def test_load_marginalpdbc_converts_hours_and_selects_last_price(tmp_path):
    price_file = tmp_path / "marginalpdbc_20230404.1"
    _write_price_file(price_file)

    prices_df = load_marginalpdbc(price_file)

    assert len(prices_df) == 24
    assert prices_df.loc[0, "hour"] == 0
    assert prices_df.loc[0, "source_hour"] == 1
    assert prices_df.loc[0, "price_1"] == 101.0
    assert prices_df.loc[0, "price_2"] == 201.0
    assert prices_df.loc[0, "grid_price"] == 201.0
    assert prices_df.loc[23, "hour"] == 23


def test_build_hourly_inputs_from_marginalpdbc(tmp_path):
    price_file = tmp_path / "marginalpdbc_20230404.1"
    _write_price_file(price_file)

    hourly_df = build_hourly_inputs_from_marginalpdbc(price_file, renewable_available=0.25)

    validate_hourly_inputs(hourly_df)
    assert list(hourly_df.columns) == ["hour", "renewable_available", "grid_price"]
    assert hourly_df.loc[0, "renewable_available"] == 0.25
    assert hourly_df.loc[0, "grid_price"] == 201.0


def test_update_hourly_inputs_prices_preserves_renewable_profile(tmp_path):
    price_file = tmp_path / "marginalpdbc_20230404.1"
    _write_price_file(price_file)
    hourly_inputs = pd.DataFrame(
        {
            "hour": list(range(24)),
            "renewable_available": [hour / 1000 for hour in range(24)],
            "grid_price": [0.0] * 24,
        }
    )

    updated_df = update_hourly_inputs_prices(hourly_inputs, price_file, price_column="first")

    validate_hourly_inputs(updated_df)
    assert updated_df.loc[0, "renewable_available"] == 0.0
    assert updated_df.loc[23, "renewable_available"] == 0.023
    assert updated_df.loc[0, "grid_price"] == 101.0
    assert updated_df.loc[23, "grid_price"] == 124.0


def test_write_hourly_inputs_from_marginalpdbc_creates_file(tmp_path):
    price_file = tmp_path / "marginalpdbc_20230404.1"
    output = tmp_path / "hourly_inputs.csv"
    _write_price_file(price_file)

    hourly_df = write_hourly_inputs_from_marginalpdbc(price_file, output)

    assert output.exists()
    written = pd.read_csv(output)
    assert len(written) == len(hourly_df)
    assert list(written.columns) == ["hour", "renewable_available", "grid_price"]
