from pathlib import Path

import pandas as pd

from src.data.solar_profiles import (
    build_daily_hourly_inputs,
    build_daily_hourly_solar_profile,
    build_monthly_hourly_inputs,
    build_monthly_hourly_solar_profile,
    load_pvgis_timeseries,
    update_hourly_inputs_daily_solar,
    update_hourly_inputs_solar,
    write_monthly_hourly_solar_profile,
)


def _write_pvgis_file(path: Path) -> None:
    rows = [
        "Latitude (decimal degrees):\t40.440",
        "Longitude (decimal degrees):\t-3.680",
        "Nominal power of the PV system (c-Si) (kWp):\t250.0",
        "time,P,Gb(i),Gd(i),Gr(i),H_sun,T2m,WS10m,Int",
    ]
    for month in range(1, 13):
        for day in (1, 2):
            for hour in range(24):
                power = month * 1000 + hour * 10
                rows.append(f"2023{month:02d}{day:02d}:{hour:02d}10,{power},0,0,0,0,0,0,0")
    rows.extend(["", "P: PV system power (W)"])
    path.write_text("\n".join(rows), encoding="utf-8")


def test_load_pvgis_timeseries_normalizes_power_and_time(tmp_path):
    source = tmp_path / "pvgis.csv"
    _write_pvgis_file(source)

    solar_df = load_pvgis_timeseries(source, target_capacity_kwp=500)

    assert len(solar_df) == 576
    assert solar_df.loc[0, "month"] == 1
    assert solar_df.loc[0, "hour"] == 0
    assert solar_df.loc[0, "power_w"] == 2000.0
    assert solar_df.loc[0, "power_mw"] == 0.002


def test_build_monthly_hourly_solar_profile(tmp_path):
    source = tmp_path / "pvgis.csv"
    _write_pvgis_file(source)

    profile_df = build_monthly_hourly_solar_profile(source)

    assert len(profile_df) == 288
    assert profile_df.loc[(profile_df.month == 4) & (profile_df.hour == 12), "renewable_available"].iloc[0] == 0.00412
    assert profile_df.loc[(profile_df.month == 4) & (profile_df.hour == 12), "sample_count"].iloc[0] == 2


def test_build_monthly_hourly_inputs_and_update_existing_prices(tmp_path):
    source = tmp_path / "pvgis.csv"
    _write_pvgis_file(source)
    existing = pd.DataFrame(
        {
            "hour": list(range(24)),
            "renewable_available": [0.0] * 24,
            "grid_price": [100.0 + hour for hour in range(24)],
        }
    )
    profile_df = build_monthly_hourly_solar_profile(source)

    hourly_df = build_monthly_hourly_inputs(source, 6, grid_price=50)
    updated_df = update_hourly_inputs_solar(existing, profile_df, 6)

    assert list(hourly_df.columns) == ["hour", "renewable_available", "grid_price"]
    assert hourly_df.loc[0, "grid_price"] == 50.0
    assert updated_df.loc[12, "grid_price"] == 112.0
    assert updated_df.loc[12, "renewable_available"] == 0.00612


def test_build_daily_hourly_profile_and_update_existing_prices(tmp_path):
    source = tmp_path / "pvgis.csv"
    _write_pvgis_file(source)
    existing = pd.DataFrame(
        {
            "hour": list(range(24)),
            "renewable_available": [0.0] * 24,
            "grid_price": [100.0 + hour for hour in range(24)],
        }
    )

    profile_df = build_daily_hourly_solar_profile(source, "2023-04-02")
    hourly_df = build_daily_hourly_inputs(source, "2023-04-02", grid_price=50)
    updated_df = update_hourly_inputs_daily_solar(existing, source, "2023-04-02")

    assert len(profile_df) == 24
    assert profile_df.loc[12, "renewable_available"] == 0.00412
    assert hourly_df.loc[12, "grid_price"] == 50.0
    assert updated_df.loc[12, "grid_price"] == 112.0
    assert updated_df.loc[12, "renewable_available"] == 0.00412


def test_write_monthly_hourly_solar_profile_creates_file(tmp_path):
    source = tmp_path / "pvgis.csv"
    output = tmp_path / "monthly_solar.csv"
    _write_pvgis_file(source)

    profile_df = write_monthly_hourly_solar_profile(source, output)

    assert output.exists()
    written = pd.read_csv(output)
    assert len(written) == len(profile_df)
