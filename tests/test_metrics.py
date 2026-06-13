from src.data.synthetic import generate_toy_config
from src.evaluation.metrics import compute_summary_metrics
import pandas as pd


def test_compute_summary_metrics():
    """Check that the summary metrics aggregate energy and cost correctly.

    The fixture is intentionally tiny so the expected values remain easy to
    verify by inspection and stable across refactors.
    """

    config = generate_toy_config()
    hourly_results = pd.DataFrame(
        {
            "total_load": [10.0, 20.0],
            "baseline_load": [1.0, 1.0],
            "flexible_load": [9.0, 19.0],
            "renewable_available": [5.0, 6.0],
            "renewable_consumption": [4.0, 5.0],
            "renewable_curtailment": [1.0, 1.0],
            "grid_consumption": [6.0, 15.0],
            "grid_price": [100.0, 200.0],
        }
    )

    metrics = compute_summary_metrics(hourly_results, config)

    assert metrics["peak_load"] == 20.0
    assert metrics["baseline_energy_mwh"] == 2.0
    assert metrics["flexible_energy_mwh"] == 28.0
    assert metrics["renewable_available_mwh"] == 11.0
    assert metrics["renewable_energy_mwh"] == 9.0
    assert metrics["renewable_curtailment_mwh"] == 2.0
    assert metrics["grid_energy_mwh"] == 21.0
    assert metrics["grid_cost"] == 3600.0
    assert metrics["renewable_cost"] == 550.0
    assert metrics["contracted_power"] == 0.2
    assert metrics["peak_over_contracted"] == 19.8
    assert metrics["peak_cost"] == 20000.0
    assert metrics["total_cost"] == 24150.0
