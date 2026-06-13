from pathlib import Path

import pandas as pd

from src.data.job_instances import load_job_instance_csv, write_imported_job_instance
from src.data.validation import validate_jobs


def _write_source(path: Path) -> None:
    source = pd.DataFrame(
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
            },
            {
                "job_id": "TR-01",
                "tier": "training",
                "gpus": 8,
                "e_kw": 6.5,
                "duration_h": 8,
                "t_min": 8,
                "t_max": 8,
                "delta": 16,
                "alpha_B": 1,
                "alpha_C": 1,
                "alpha_D": 0,
                "n_start_slots": 1,
                "qubo_vars": 2,
            },
        ]
    )
    source.to_csv(path, index=False)


def test_load_job_instance_csv_normalizes_shared_schema(tmp_path):
    source = tmp_path / "jobs_tense.csv"
    _write_source(source)

    jobs_df = load_job_instance_csv(source)

    validate_jobs(jobs_df)
    assert list(jobs_df["job_id"]) == ["FT-01", "TR-01"]
    assert list(jobs_df["category"]) == ["fine_tuning", "training"]
    assert list(jobs_df["gpus"]) == [4, 8]
    assert list(jobs_df["duration"]) == [4, 8]
    assert list(jobs_df["earliest_start"]) == [9, 7]
    assert list(jobs_df["latest_start"]) == [11, 7]
    assert jobs_df.loc[0, "power"] == 0.0032
    assert jobs_df.loc[1, "alpha_D"] == 0


def test_write_imported_job_instance_creates_output_file(tmp_path):
    source = tmp_path / "jobs_light.csv"
    output = tmp_path / "processed" / "jobs.csv"
    _write_source(source)

    jobs_df = write_imported_job_instance(source, output)

    assert output.exists()
    written = pd.read_csv(output)
    assert len(written) == len(jobs_df)
    assert {"job_id", "category", "gpus", "power", "earliest_start", "latest_start"}.issubset(written.columns)
