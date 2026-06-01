from __future__ import annotations

import pandas as pd
from typer.testing import CliRunner

from cp202611.cli import app
from cp202611.evidence import build_evidence_package


runner = CliRunner()


def test_build_evidence_package_summarizes_available_outputs(tmp_path):
    acceptance_dir = tmp_path / "acceptance"
    robustness_dir = tmp_path / "robustness"
    output_dir = tmp_path / "evidence"
    acceptance_dir.mkdir()
    robustness_dir.mkdir()

    pd.DataFrame(
        [
            {"check": "validation_errors_zero", "passed": True, "value": 0, "threshold": 0},
            {"check": "stress_all_converged", "passed": True, "value": 7, "threshold": 7},
        ]
    ).to_csv(acceptance_dir / "acceptance_summary.csv", index=False)
    pd.DataFrame(
        [
            {
                "case_id": "base",
                "passed": True,
                "converged": True,
                "cost_delta_pct": 0.0,
                "carbon_delta_pct": 0.0,
                "exergy_delta_pct": 0.0,
                "max_comfort_slack_c": 0.0,
                "total_unmet_mid_mwh": 0.0,
            },
            {
                "case_id": "cold_minus_5c",
                "passed": True,
                "converged": True,
                "cost_delta_pct": 32.3,
                "carbon_delta_pct": 26.9,
                "exergy_delta_pct": 80.6,
                "max_comfort_slack_c": 0.0,
                "total_unmet_mid_mwh": 0.0,
            },
        ]
    ).to_csv(robustness_dir / "robustness_matrix.csv", index=False)

    result = build_evidence_package(
        output_dir=output_dir,
        acceptance_dir=acceptance_dir,
        robustness_dir=robustness_dir,
        dataset_label="synthetic_week",
    )

    report = result.report_path.read_text(encoding="utf-8")
    assert result.report_path.exists()
    assert result.overall_ready
    assert "评分点对齐" in report
    assert "validation_errors_zero" in report
    assert "cold_minus_5c" in report
    assert "官方数据" in report


def test_evidence_package_cli_writes_report(tmp_path):
    acceptance_dir = tmp_path / "acceptance"
    robustness_dir = tmp_path / "robustness"
    output_dir = tmp_path / "evidence_cli"
    acceptance_dir.mkdir()
    robustness_dir.mkdir()
    pd.DataFrame([{"check": "validation_errors_zero", "passed": True, "value": 0, "threshold": 0}]).to_csv(
        acceptance_dir / "acceptance_summary.csv",
        index=False,
    )
    pd.DataFrame(
        [
            {
                "case_id": "base",
                "passed": True,
                "converged": True,
                "cost_delta_pct": 0.0,
                "carbon_delta_pct": 0.0,
                "exergy_delta_pct": 0.0,
                "max_comfort_slack_c": 0.0,
                "total_unmet_mid_mwh": 0.0,
            }
        ]
    ).to_csv(robustness_dir / "robustness_matrix.csv", index=False)

    result = runner.invoke(
        app,
        [
            "evidence-package",
            "--acceptance-dir",
            str(acceptance_dir),
            "--robustness-dir",
            str(robustness_dir),
            "--output-dir",
            str(output_dir),
            "--dataset-label",
            "synthetic_test",
        ],
    )

    assert result.exit_code == 0
    assert "Evidence package" in result.output
    assert (output_dir / "evidence_package.md").exists()
