from __future__ import annotations

from typer.testing import CliRunner

from cp202611.cli import app
from cp202611.dataio import write_processed_dataset
from cp202611.robustness import run_robustness_matrix
from cp202611.stress import StressCase
from cp202611.synthetic import create_synthetic_season


runner = CliRunner()


def test_robustness_matrix_writes_report_and_deltas(tmp_path):
    scenario = create_synthetic_season(n_hours=72)
    output_dir = tmp_path / "robustness"

    result = run_robustness_matrix(
        scenario=scenario,
        output_dir=output_dir,
        cases=[
            StressCase(case_id="base"),
            StressCase(case_id="cold_minus_1c", outdoor_delta_c=-1.0),
        ],
        n_typical_days=3,
        max_iterations=1,
    )

    assert result.passed
    assert len(result.matrix) == 2
    assert {"cost_delta_pct", "carbon_delta_pct", "exergy_delta_pct", "passed"}.issubset(result.matrix.columns)
    assert bool(result.matrix.loc[result.matrix["case_id"] == "base", "passed"].iloc[0])
    assert (output_dir / "robustness_matrix.csv").exists()
    assert (output_dir / "robustness_report.md").exists()


def test_robustness_matrix_cli_runs_quick_mode(tmp_path):
    scenario = create_synthetic_season(n_hours=72)
    dataset_dir = tmp_path / "dataset"
    output_dir = tmp_path / "robustness_cli"
    write_processed_dataset(scenario, dataset_dir)

    result = runner.invoke(
        app,
        [
            "robustness-matrix",
            "--dataset-dir",
            str(dataset_dir),
            "--output-dir",
            str(output_dir),
            "--n-days",
            "3",
            "--max-iterations",
            "1",
            "--mode",
            "quick",
        ],
    )

    assert result.exit_code == 0
    assert "Overall passed: True" in result.output
    assert (output_dir / "robustness_matrix.csv").exists()
