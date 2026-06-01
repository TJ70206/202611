from __future__ import annotations

from cp202611.acceptance import run_acceptance_check
from cp202611.dataio import write_processed_dataset
from cp202611.stress import StressCase
from cp202611.synthetic import create_synthetic_season


def test_acceptance_check_writes_summary_and_report(tmp_path):
    scenario = create_synthetic_season(n_hours=72)
    dataset_dir = tmp_path / "dataset"
    output_dir = tmp_path / "acceptance"
    write_processed_dataset(scenario, dataset_dir)

    result = run_acceptance_check(
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        n_typical_days=3,
        max_iterations=1,
        carbon_prices=[0.0],
        exergy_penalties=[0.0],
        stress_cases=[StressCase(case_id="base")],
        run_visuals=False,
    )

    assert result.passed
    assert (output_dir / "acceptance_summary.csv").exists()
    assert (output_dir / "acceptance_report.md").exists()
    assert (output_dir / "benchmark" / "benchmark_summary.csv").exists()
    assert (output_dir / "stress" / "stress_results.csv").exists()
    assert (output_dir / "pareto" / "pareto_runs.csv").exists()
