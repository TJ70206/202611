from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from cp202611.analysis.diagnostics import compute_mvp_diagnostics
from cp202611.benchmark import BenchmarkBundle, run_planning_benchmark_bundle
from cp202611.dataio import load_processed_dataset
from cp202611.pareto import ParetoScanResult, run_pareto_scan
from cp202611.schema import MVPScenario
from cp202611.stress import DEFAULT_STRESS_CASES, StressCase, run_stress_suite
from cp202611.validation import ValidationReport, validate_scenario
from cp202611.visualization import create_report_figures


@dataclass(frozen=True)
class AcceptanceCriteria:
    max_balance_residual_mw: float = 1e-6
    max_comfort_slack_c: float = 1e-5
    max_unmet_mid_mwh: float = 1e-5


@dataclass(frozen=True)
class AcceptanceResult:
    passed: bool
    summary: pd.DataFrame
    validation: ValidationReport
    benchmark: BenchmarkBundle
    stress: pd.DataFrame
    pareto: ParetoScanResult | None


def run_acceptance_check(
    dataset_dir: Path,
    output_dir: Path,
    n_typical_days: int = 8,
    max_iterations: int = 4,
    carbon_prices: list[float] | None = None,
    exergy_penalties: list[float] | None = None,
    stress_cases: list[StressCase] | None = None,
    run_pareto: bool = True,
    run_visuals: bool = True,
    criteria: AcceptanceCriteria | None = None,
) -> AcceptanceResult:
    """Run the pre-official-data acceptance workflow on a processed scenario."""
    criteria = criteria or AcceptanceCriteria()
    scenario = load_processed_dataset(dataset_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    validation = validate_scenario(scenario)
    validation_dir = output_dir / "validation"
    validation_dir.mkdir(parents=True, exist_ok=True)
    validation.to_dataframe().to_csv(validation_dir / "validation_issues.csv", index=False, encoding="utf-8-sig")
    (validation_dir / "validation_report.md").write_text(validation.to_markdown(), encoding="utf-8")
    if validation.error_count > 0:
        raise ValueError(f"scenario is not usable: {validation.error_count} validation errors")

    benchmark = run_planning_benchmark_bundle(
        scenario,
        n_typical_days=n_typical_days,
        max_iterations=max_iterations,
    )
    benchmark_dir = output_dir / "benchmark"
    _write_benchmark_bundle(benchmark, benchmark_dir)

    stress = run_stress_suite(
        scenario,
        cases=stress_cases or DEFAULT_STRESS_CASES,
        n_typical_days=max(3, min(n_typical_days, 4)),
        max_iterations=max_iterations,
    )
    stress_dir = output_dir / "stress"
    stress_dir.mkdir(parents=True, exist_ok=True)
    stress.to_csv(stress_dir / "stress_results.csv", index=False, encoding="utf-8-sig")

    pareto: ParetoScanResult | None = None
    pareto_dir = output_dir / "pareto"
    if run_pareto:
        pareto = run_pareto_scan(
            scenario,
            carbon_prices=carbon_prices or [0.0, 500.0, 1500.0],
            exergy_penalties=exergy_penalties or [0.0, 260.0],
            n_typical_days=max(3, min(n_typical_days, 4)),
            max_iterations=max_iterations,
        )
        pareto_dir.mkdir(parents=True, exist_ok=True)
        pareto.runs.to_csv(pareto_dir / "pareto_runs.csv", index=False, encoding="utf-8-sig")
        pareto.pareto_front.to_csv(pareto_dir / "pareto_front.csv", index=False, encoding="utf-8-sig")

    if run_visuals:
        create_report_figures(
            result_dir=benchmark_dir,
            dataset_dir=dataset_dir,
            pareto_dir=pareto_dir if pareto is not None else None,
            output_dir=output_dir / "figures",
        )

    summary = _build_acceptance_summary(
        scenario=scenario,
        validation=validation,
        benchmark=benchmark,
        stress=stress,
        pareto=pareto,
        criteria=criteria,
    )
    passed = bool(summary["passed"].all())
    summary.to_csv(output_dir / "acceptance_summary.csv", index=False, encoding="utf-8-sig")
    (output_dir / "acceptance_report.md").write_text(
        _summary_to_markdown(summary, passed),
        encoding="utf-8",
    )
    return AcceptanceResult(
        passed=passed,
        summary=summary,
        validation=validation,
        benchmark=benchmark,
        stress=stress,
        pareto=pareto,
    )


def _write_benchmark_bundle(bundle: BenchmarkBundle, output_dir: Path) -> None:
    final = bundle.planning.final_result
    output_dir.mkdir(parents=True, exist_ok=True)
    bundle.summary.to_dataframe().to_csv(output_dir / "benchmark_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([iteration.__dict__ for iteration in bundle.planning.iterations]).to_csv(
        output_dir / "feedback_iterations.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(
        {
            "day": final.typical_days.selected_days,
            "weight": [final.typical_days.weights[d] for d in final.typical_days.selected_days],
        }
    ).to_csv(output_dir / "final_typical_days.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([final.typical_days.metrics]).to_csv(
        output_dir / "final_typical_day_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    final.planning_result.source_capacity.to_csv(output_dir / "planning_capacity.csv", index=False, encoding="utf-8-sig")
    final.planning_result.spatial_assignment.to_csv(
        output_dir / "planning_spatial_assignment.csv",
        index=False,
        encoding="utf-8-sig",
    )
    final.verification_result.dispatch.to_csv(output_dir / "verification_dispatch.csv", index=False, encoding="utf-8-sig")
    final.verification_result.storage.to_csv(output_dir / "verification_storage.csv", index=False, encoding="utf-8-sig")
    final.verification_result.indoor_temperature.to_csv(
        output_dir / "verification_indoor_temperature.csv",
        index=False,
        encoding="utf-8-sig",
    )
    final.verification_result.network_edges.to_csv(output_dir / "verification_network_edges.csv", index=False, encoding="utf-8-sig")
    final.verification_result.source_capacity.to_csv(output_dir / "verification_capacity.csv", index=False, encoding="utf-8-sig")


def _build_acceptance_summary(
    scenario: MVPScenario,
    validation: ValidationReport,
    benchmark: BenchmarkBundle,
    stress: pd.DataFrame,
    pareto: ParetoScanResult | None,
    criteria: AcceptanceCriteria,
) -> pd.DataFrame:
    diagnostics = compute_mvp_diagnostics(benchmark.planning.final_result.verification_result)
    benchmark_summary = benchmark.summary
    rows = [
        _row("validation_errors_zero", validation.error_count == 0, validation.error_count, 0),
        _row("benchmark_converged", benchmark_summary.converged, benchmark_summary.converged, True),
        _row("low_balance_residual", diagnostics.max_low_balance_abs_mw <= criteria.max_balance_residual_mw, diagnostics.max_low_balance_abs_mw, criteria.max_balance_residual_mw),
        _row("mid_balance_residual", diagnostics.max_mid_balance_abs_mw <= criteria.max_balance_residual_mw, diagnostics.max_mid_balance_abs_mw, criteria.max_balance_residual_mw),
        _row("comfort_slack", benchmark_summary.max_comfort_slack_c <= criteria.max_comfort_slack_c, benchmark_summary.max_comfort_slack_c, criteria.max_comfort_slack_c),
        _row("unmet_mid_heat", benchmark_summary.total_unmet_mid_mwh <= criteria.max_unmet_mid_mwh, benchmark_summary.total_unmet_mid_mwh, criteria.max_unmet_mid_mwh),
        _row("stress_all_solved", bool(stress["solved"].fillna(False).all()), int(stress["solved"].fillna(False).sum()), len(stress)),
        _row("stress_all_converged", bool(stress["converged"].fillna(False).all()), int(stress["converged"].fillna(False).sum()), len(stress)),
        _row("stress_no_comfort_slack", float(stress["max_comfort_slack_c"].fillna(0).max()) <= criteria.max_comfort_slack_c, float(stress["max_comfort_slack_c"].fillna(0).max()), criteria.max_comfort_slack_c),
        _row("stress_no_unmet_mid", float(stress["total_unmet_mid_mwh"].fillna(0).max()) <= criteria.max_unmet_mid_mwh, float(stress["total_unmet_mid_mwh"].fillna(0).max()), criteria.max_unmet_mid_mwh),
        _row("full_hourly_horizon_present", len(scenario.hours) >= 24 and len(scenario.hours) % 24 == 0, len(scenario.hours), "whole-day hourly horizon"),
    ]
    if pareto is not None:
        rows.extend(
            [
                _row("pareto_all_converged", bool(pareto.runs["converged"].all()), int(pareto.runs["converged"].sum()), len(pareto.runs)),
                _row("pareto_front_nonempty", not pareto.pareto_front.empty, len(pareto.pareto_front), "> 0"),
            ]
        )
    return pd.DataFrame(rows)


def _row(check: str, passed: bool, value: object, threshold: object) -> dict[str, object]:
    return {"check": check, "passed": bool(passed), "value": value, "threshold": threshold}


def _summary_to_markdown(summary: pd.DataFrame, passed: bool) -> str:
    lines = [
        "# CP202611 Acceptance Report",
        "",
        f"- overall_passed: `{passed}`",
        f"- checks: `{len(summary)}`",
        "",
        "| check | passed | value | threshold |",
        "|---|---:|---:|---:|",
    ]
    for row in summary.to_dict(orient="records"):
        lines.append(f"| {row['check']} | {row['passed']} | {row['value']} | {row['threshold']} |")
    return "\n".join(lines) + "\n"
