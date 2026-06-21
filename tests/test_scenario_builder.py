from pathlib import Path

import pandas as pd

from src.data.scenarios import build_document_clusters, build_repo_scenario, build_thesis_scenario


def _write_jobs(path: Path) -> None:
    pd.DataFrame(
        [
            {
                "job_id": "FT-01",
                "tier": "finetune",
                "gpus": 4,
                "e_kw": 3.2,
                "duration_h": 4,
                "t_min": 10,
                "t_max": 12,
                "delta": 16,
                "alpha_B": 1,
                "alpha_C": 1,
                "alpha_D": 1,
                "n_start_slots": 3,
                "qubo_vars": 9,
            }
        ]
    ).to_csv(path, index=False)


def _write_price(path: Path) -> None:
    rows = ["MARGINALPDBC;"]
    rows.extend(f"2023;04;04;{hour};{100 + hour};{200 + hour};" for hour in range(1, 25))
    rows.append("*")
    path.write_text("\n".join(rows), encoding="utf-8")


def _write_solar(path: Path) -> None:
    rows = ["time,P,Gb(i),Gd(i),Gr(i),H_sun,T2m,WS10m,Int"]
    rows.extend(f"20230404:{hour:02d}10,{hour * 1000},0,0,0,0,0,0,0" for hour in range(24))
    path.write_text("\n".join(rows), encoding="utf-8")


def _write_monthly_solar(path: Path) -> None:
    rows = [
        {
            "month": 4,
            "hour": hour,
            "renewable_available": hour / 1000,
            "avg_power_kw": hour,
            "sample_count": 30,
        }
        for hour in range(24)
    ]
    pd.DataFrame(rows).to_csv(path, index=False)


def test_build_document_clusters_excludes_inference_zone_by_default():
    clusters_df = build_document_clusters()

    assert list(clusters_df["cluster_id"]) == ["cluster_b", "cluster_c", "cluster_d"]
    assert list(clusters_df["gpu_capacity"]) == [120, 160, 32]


def test_build_thesis_scenario_combines_source_files(tmp_path):
    jobs = tmp_path / "jobs.csv"
    price = tmp_path / "marginalpdbc.1"
    solar = tmp_path / "solar.csv"
    _write_jobs(jobs)
    _write_price(price)
    _write_solar(solar)

    jobs_df, hourly_df, clusters_df, config = build_thesis_scenario(
        jobs_csv=jobs,
        price_file=price,
        solar_csv=solar,
        scenario_date="2023-04-04",
        baseline_load_mw=0.01,
        renewable_price=40,
        peak_price=500,
        contracted_power=0.222,
    )

    assert len(jobs_df) == 1
    assert len(hourly_df) == 24
    assert len(clusters_df) == 3
    assert hourly_df.loc[0, "grid_price"] == 201.0
    assert hourly_df.loc[12, "renewable_available"] == 0.012
    assert (hourly_df["baseline_load"] == 0.01).all()
    assert config.renewable_price == 40


def test_build_repo_scenario_accepts_explicit_job_instance(tmp_path):
    project_root = tmp_path
    jobs = project_root / "jobs.csv"
    price = project_root / "docs" / "energy_price" / "marginalpdbc_20230404.1"
    solar = project_root / "data" / "solar_profile" / "monthly_solar_profiles.csv"
    price.parent.mkdir(parents=True)
    solar.parent.mkdir(parents=True)
    _write_jobs(jobs)
    _write_price(price)
    _write_monthly_solar(solar)

    jobs_df, hourly_df, clusters_df, config = build_repo_scenario(
        project_root=project_root,
        jobs_csv=jobs,
        energy_scenario="clear_sky",
        solar_profile_csv=solar,
        baseline_load_mw=0.02,
        renewable_price=30,
        peak_price=900,
        contracted_power=0.1,
    )

    assert len(jobs_df) == 1
    assert len(hourly_df) == 24
    assert len(clusters_df) == 3
    assert hourly_df.loc[0, "grid_price"] == 201.0
    assert hourly_df.loc[12, "renewable_available"] == 0.012
    assert (hourly_df["baseline_load"] == 0.02).all()
    assert config.renewable_price == 30


def test_build_repo_scenario_enables_battery_only_when_requested(tmp_path):
    project_root = tmp_path
    jobs = project_root / "jobs.csv"
    price = project_root / "docs" / "energy_price" / "marginalpdbc_20230404.1"
    solar = project_root / "data" / "solar_profile" / "monthly_solar_profiles.csv"
    price.parent.mkdir(parents=True)
    solar.parent.mkdir(parents=True)
    _write_jobs(jobs)
    _write_price(price)
    _write_monthly_solar(solar)

    _, _, _, config_without_battery = build_repo_scenario(
        project_root=project_root,
        jobs_csv=jobs,
        energy_scenario="clear_sky",
        solar_profile_csv=solar,
        battery_power_capacity=0.5,
        battery_energy_capacity=1.0,
    )
    _, _, _, config_with_battery = build_repo_scenario(
        project_root=project_root,
        jobs_csv=jobs,
        energy_scenario="clear_sky",
        solar_profile_csv=solar,
        use_battery=True,
        battery_power_capacity=0.5,
        battery_energy_capacity=1.0,
        battery_initial_soc=0.2,
        battery_final_soc=0.1,
    )

    assert config_without_battery.battery_power_capacity == 0.0
    assert config_without_battery.battery_energy_capacity == 0.0
    assert config_with_battery.battery_power_capacity == 0.5
    assert config_with_battery.battery_energy_capacity == 1.0
    assert config_with_battery.battery_initial_soc == 0.2
    assert config_with_battery.battery_final_soc == 0.1
