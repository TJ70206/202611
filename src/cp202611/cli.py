from __future__ import annotations

from pathlib import Path

import pandas as pd
import typer
from rich.console import Console
from rich.table import Table

from cp202611.acceptance import run_acceptance_check
from cp202611.analysis.diagnostics import compute_mvp_diagnostics
from cp202611.adapters.official import preprocess_official_data
from cp202611.benchmark import run_planning_benchmark_bundle
from cp202611.dataio import load_processed_dataset, write_processed_dataset
from cp202611.evidence import build_evidence_package
from cp202611.optimization.mvp_model import solve_mvp
from cp202611.pareto import run_pareto_scan
from cp202611.planning import run_feedback_corrected_planning, run_two_stage_planning
from cp202611.robustness import run_robustness_matrix
from cp202611.stress import StressCase, run_stress_suite
from cp202611.synthetic import create_synthetic_mvp, create_synthetic_season, create_synthetic_week
from cp202611.typical_days import select_peak_preserving_kmedoids
from cp202611.validation import validate_scenario
from cp202611.visualization import create_report_figures

app = typer.Typer(help="CP-202611 planning model command line.")
console = Console()


def _parse_float_list(value: str) -> list[float]:
    try:
        parsed = [float(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as exc:
        raise typer.BadParameter("expected a comma-separated numeric list") from exc
    if not parsed:
        raise typer.BadParameter("list must not be empty")
    return parsed


@app.command()
def mvp(output: Path = typer.Option(Path("outputs/mvp_dispatch.csv"), help="Dispatch CSV output path.")) -> None:
    """Run the strong core MVP with synthetic data."""
    scenario = create_synthetic_mvp()
    result = solve_mvp(scenario)
    diagnostics = compute_mvp_diagnostics(result)

    output.parent.mkdir(parents=True, exist_ok=True)
    result.dispatch.to_csv(output, index=False, encoding="utf-8-sig")
    result.indoor_temperature.to_csv(output.with_name("mvp_indoor_temperature.csv"), index=False, encoding="utf-8-sig")
    result.storage.to_csv(output.with_name("mvp_storage.csv"), index=False, encoding="utf-8-sig")
    result.spatial_assignment.to_csv(output.with_name("mvp_spatial_assignment.csv"), index=False, encoding="utf-8-sig")
    result.network_edges.to_csv(output.with_name("mvp_network_edges.csv"), index=False, encoding="utf-8-sig")
    result.source_capacity.to_csv(output.with_name("mvp_capacity.csv"), index=False, encoding="utf-8-sig")

    table = Table(title="CP202611 MVP Solve Summary")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("termination", result.termination_condition)
    table.add_row("objective", f"{result.objective_value:,.2f}")
    table.add_row("low balance max abs MW", f"{diagnostics.max_low_balance_abs_mw:.3e}")
    table.add_row("mid balance max abs MW", f"{diagnostics.max_mid_balance_abs_mw:.3e}")
    table.add_row("SOC closure abs MWh", f"{diagnostics.soc_closure_abs_mwh:.3e}")
    table.add_row("max comfort slack C", f"{diagnostics.max_comfort_slack_c:.3e}")
    table.add_row("network loss MWh", f"{diagnostics.network_loss_mwh:.3f}")
    table.add_row("offpeak charge MWh", f"{diagnostics.storage_charged_in_offpeak_mwh:.3f}")
    table.add_row("peak discharge MWh", f"{diagnostics.storage_discharged_in_peak_mwh:.3f}")
    console.print(table)
    console.print(f"Wrote dispatch to {output}")


@app.command("generate-synthetic")
def generate_synthetic(
    output_dir: Path = typer.Option(Path("data/processed/synthetic_week"), help="Canonical processed dataset directory."),
    hours: int = typer.Option(168, min=24, help="Synthetic scenario horizon in hours."),
    profile: str = typer.Option("week", help="Synthetic profile: week or season."),
) -> None:
    """Generate a canonical synthetic processed dataset."""
    if profile == "week":
        scenario = create_synthetic_mvp(n_hours=hours)
    elif profile == "season":
        scenario = create_synthetic_season(n_hours=hours)
    else:
        raise typer.BadParameter("profile must be one of: week, season")
    write_processed_dataset(scenario, output_dir)
    console.print(f"Wrote canonical processed dataset to {output_dir}")


@app.command("solve")
def solve(
    dataset_dir: Path = typer.Option(Path("data/processed/synthetic_week"), help="Canonical processed dataset directory."),
    output: Path = typer.Option(Path("outputs/solve_dispatch.csv"), help="Dispatch CSV output path."),
) -> None:
    """Solve a canonical processed scenario."""
    scenario = load_processed_dataset(dataset_dir)
    result = solve_mvp(scenario)
    diagnostics = compute_mvp_diagnostics(result)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.dispatch.to_csv(output, index=False, encoding="utf-8-sig")
    result.indoor_temperature.to_csv(output.with_name("solve_indoor_temperature.csv"), index=False, encoding="utf-8-sig")
    result.storage.to_csv(output.with_name("solve_storage.csv"), index=False, encoding="utf-8-sig")
    result.spatial_assignment.to_csv(output.with_name("solve_spatial_assignment.csv"), index=False, encoding="utf-8-sig")
    result.network_edges.to_csv(output.with_name("solve_network_edges.csv"), index=False, encoding="utf-8-sig")
    result.source_capacity.to_csv(output.with_name("solve_capacity.csv"), index=False, encoding="utf-8-sig")
    console.print(f"Solved {scenario.scenario_id}: {result.termination_condition}, objective={result.objective_value:,.2f}")
    console.print(f"Max heat balance residual: low={diagnostics.max_low_balance_abs_mw:.3e}, mid={diagnostics.max_mid_balance_abs_mw:.3e}")
    console.print(f"Network loss: {diagnostics.network_loss_mwh:.3f} MWh")


@app.command("validate-data")
def validate_data(
    dataset_dir: Path = typer.Option(Path("data/processed/synthetic_week"), help="Canonical processed dataset directory."),
    output_dir: Path = typer.Option(Path("outputs/validation"), help="Validation output directory."),
) -> None:
    """Validate a canonical processed scenario before optimization."""
    scenario = load_processed_dataset(dataset_dir)
    report = validate_scenario(scenario)
    output_dir.mkdir(parents=True, exist_ok=True)
    report.to_dataframe().to_csv(output_dir / "validation_issues.csv", index=False, encoding="utf-8-sig")
    (output_dir / "validation_report.md").write_text(report.to_markdown(), encoding="utf-8")

    table = Table(title="CP202611 Data Validation Summary")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("usable", str(report.is_usable))
    table.add_row("errors", str(report.error_count))
    table.add_row("warnings", str(report.warning_count))
    console.print(table)
    console.print(f"Wrote validation outputs to {output_dir}")


@app.command("preprocess-official")
def preprocess_official(
    raw_dir: Path = typer.Option(Path("data/raw/official"), help="Organizer/public raw data directory."),
    mapping: Path = typer.Option(Path("configs/field_mapping_official_template.yaml"), help="YAML field mapping file."),
    output_dir: Path = typer.Option(Path("data/processed/official"), help="Canonical processed dataset directory."),
) -> None:
    """Map raw organizer/public tables to the canonical processed-data contract."""
    report = preprocess_official_data(raw_dir=raw_dir, mapping_path=mapping, output_dir=output_dir)

    table = Table(title="CP202611 Official Data Preprocess Summary")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("usable", str(report.is_usable))
    table.add_row("errors", str(report.error_count))
    table.add_row("warnings", str(report.warning_count))
    console.print(table)
    console.print(f"Wrote canonical processed dataset and validation report to {output_dir}")


@app.command("two-stage")
def two_stage(
    dataset_dir: Path = typer.Option(Path("data/processed/synthetic_week"), help="Canonical full-horizon dataset directory."),
    n_days: int = typer.Option(4, min=3, help="Number of representative days for capacity planning."),
    output_dir: Path = typer.Option(Path("outputs/two_stage"), help="Two-stage output directory."),
) -> None:
    """Run typical-day capacity planning and full-horizon fixed-capacity verification."""
    scenario = load_processed_dataset(dataset_dir)
    result = run_two_stage_planning(scenario, n_typical_days=n_days)
    planning_diag = compute_mvp_diagnostics(result.planning_result)
    verification_diag = compute_mvp_diagnostics(result.verification_result)

    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "day": result.typical_days.selected_days,
            "weight": [result.typical_days.weights[d] for d in result.typical_days.selected_days],
        }
    ).to_csv(output_dir / "typical_days.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([result.typical_days.metrics]).to_csv(output_dir / "typical_day_metrics.csv", index=False, encoding="utf-8-sig")

    result.planning_result.source_capacity.to_csv(output_dir / "planning_capacity.csv", index=False, encoding="utf-8-sig")
    result.planning_result.spatial_assignment.to_csv(output_dir / "planning_spatial_assignment.csv", index=False, encoding="utf-8-sig")
    result.verification_result.dispatch.to_csv(output_dir / "verification_dispatch.csv", index=False, encoding="utf-8-sig")
    result.verification_result.storage.to_csv(output_dir / "verification_storage.csv", index=False, encoding="utf-8-sig")
    result.verification_result.indoor_temperature.to_csv(output_dir / "verification_indoor_temperature.csv", index=False, encoding="utf-8-sig")
    result.verification_result.network_edges.to_csv(output_dir / "verification_network_edges.csv", index=False, encoding="utf-8-sig")
    result.verification_result.source_capacity.to_csv(output_dir / "verification_capacity.csv", index=False, encoding="utf-8-sig")

    table = Table(title="CP202611 Two-stage Planning Summary")
    table.add_column("Metric")
    table.add_column("Planning", justify="right")
    table.add_column("Verification", justify="right")
    table.add_row("termination", result.planning_result.termination_condition, result.verification_result.termination_condition)
    table.add_row("objective", f"{result.planning_result.objective_value:,.2f}", f"{result.verification_result.objective_value:,.2f}")
    table.add_row("low residual MW", f"{planning_diag.max_low_balance_abs_mw:.3e}", f"{verification_diag.max_low_balance_abs_mw:.3e}")
    table.add_row("mid residual MW", f"{planning_diag.max_mid_balance_abs_mw:.3e}", f"{verification_diag.max_mid_balance_abs_mw:.3e}")
    table.add_row("comfort slack C", f"{planning_diag.max_comfort_slack_c:.3e}", f"{verification_diag.max_comfort_slack_c:.3e}")
    table.add_row("network loss MWh", f"{planning_diag.network_loss_mwh:.3f}", f"{verification_diag.network_loss_mwh:.3f}")
    console.print(table)
    console.print(f"Wrote two-stage outputs to {output_dir}")


@app.command("feedback-plan")
def feedback_plan(
    dataset_dir: Path = typer.Option(Path("data/processed/synthetic_week"), help="Canonical full-horizon dataset directory."),
    n_days: int = typer.Option(4, min=3, help="Initial number of representative days."),
    max_iterations: int = typer.Option(4, min=1, help="Maximum feedback-correction iterations."),
    output_dir: Path = typer.Option(Path("outputs/feedback_plan"), help="Feedback-planning output directory."),
) -> None:
    """Run typical-day planning with worst-day feedback correction."""
    scenario = load_processed_dataset(dataset_dir)
    result = run_feedback_corrected_planning(
        scenario,
        n_typical_days=n_days,
        max_iterations=max_iterations,
    )
    final = result.final_result
    planning_diag = compute_mvp_diagnostics(final.planning_result)
    verification_diag = compute_mvp_diagnostics(final.verification_result)

    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([iteration.__dict__ for iteration in result.iterations]).to_csv(
        output_dir / "feedback_iterations.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(
        {
            "day": final.typical_days.selected_days,
            "weight": [final.typical_days.weights[d] for d in final.typical_days.selected_days],
        }
    ).to_csv(output_dir / "final_typical_days.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([final.typical_days.metrics]).to_csv(output_dir / "final_typical_day_metrics.csv", index=False, encoding="utf-8-sig")

    final.planning_result.source_capacity.to_csv(output_dir / "planning_capacity.csv", index=False, encoding="utf-8-sig")
    final.planning_result.spatial_assignment.to_csv(output_dir / "planning_spatial_assignment.csv", index=False, encoding="utf-8-sig")
    final.verification_result.dispatch.to_csv(output_dir / "verification_dispatch.csv", index=False, encoding="utf-8-sig")
    final.verification_result.storage.to_csv(output_dir / "verification_storage.csv", index=False, encoding="utf-8-sig")
    final.verification_result.indoor_temperature.to_csv(output_dir / "verification_indoor_temperature.csv", index=False, encoding="utf-8-sig")
    final.verification_result.network_edges.to_csv(output_dir / "verification_network_edges.csv", index=False, encoding="utf-8-sig")
    final.verification_result.source_capacity.to_csv(output_dir / "verification_capacity.csv", index=False, encoding="utf-8-sig")

    table = Table(title="CP202611 Feedback-corrected Planning Summary")
    table.add_column("Metric")
    table.add_column("Planning", justify="right")
    table.add_column("Verification", justify="right")
    table.add_row("converged", str(result.converged), str(result.converged))
    table.add_row("iterations", str(len(result.iterations)), str(len(result.iterations)))
    table.add_row("termination", final.planning_result.termination_condition, final.verification_result.termination_condition)
    table.add_row("objective", f"{final.planning_result.objective_value:,.2f}", f"{final.verification_result.objective_value:,.2f}")
    table.add_row("low residual MW", f"{planning_diag.max_low_balance_abs_mw:.3e}", f"{verification_diag.max_low_balance_abs_mw:.3e}")
    table.add_row("mid residual MW", f"{planning_diag.max_mid_balance_abs_mw:.3e}", f"{verification_diag.max_mid_balance_abs_mw:.3e}")
    table.add_row("comfort slack C", f"{planning_diag.max_comfort_slack_c:.3e}", f"{verification_diag.max_comfort_slack_c:.3e}")
    table.add_row("unmet mid MWh", f"{planning_diag.total_unmet_mid_mwh:.3e}", f"{verification_diag.total_unmet_mid_mwh:.3e}")
    table.add_row("network loss MWh", f"{planning_diag.network_loss_mwh:.3f}", f"{verification_diag.network_loss_mwh:.3f}")
    console.print(table)
    console.print(f"Wrote feedback-planning outputs to {output_dir}")


@app.command("pareto-scan")
def pareto_scan(
    dataset_dir: Path = typer.Option(Path("data/processed/synthetic_week"), help="Canonical full-horizon dataset directory."),
    carbon_prices: str = typer.Option("0,80,160", help="Comma-separated carbon prices in CNY/t."),
    exergy_penalties: str = typer.Option("0,260", help="Comma-separated exergy penalties in CNY/MWh-eq."),
    n_days: int = typer.Option(4, min=3, help="Initial representative-day count."),
    max_iterations: int = typer.Option(4, min=1, help="Maximum feedback-correction iterations for each run."),
    output_dir: Path = typer.Option(Path("outputs/pareto"), help="Pareto output directory."),
) -> None:
    """Run carbon/exergy weighted scans and export Pareto-front candidates."""
    scenario = load_processed_dataset(dataset_dir)
    carbon_price_values = _parse_float_list(carbon_prices)
    exergy_penalty_values = _parse_float_list(exergy_penalties)
    result = run_pareto_scan(
        scenario,
        carbon_prices=carbon_price_values,
        exergy_penalties=exergy_penalty_values,
        n_typical_days=n_days,
        max_iterations=max_iterations,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    result.runs.to_csv(output_dir / "pareto_runs.csv", index=False, encoding="utf-8-sig")
    result.pareto_front.to_csv(output_dir / "pareto_front.csv", index=False, encoding="utf-8-sig")

    table = Table(title="CP202611 Pareto Scan Summary")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("runs", str(len(result.runs)))
    table.add_row("pareto candidates", str(len(result.pareto_front)))
    table.add_row("all converged", str(bool(result.runs["converged"].all())))
    table.add_row("min economic cost CNY", f"{result.runs['economic_cost_cny'].min():,.2f}")
    table.add_row("min carbon emissions t", f"{result.runs['carbon_emissions_t'].min():,.3f}")
    table.add_row("min exergy loss MWh-eq", f"{result.runs['exergy_loss_mwh_eq'].min():,.3f}")
    console.print(table)
    console.print(f"Wrote Pareto outputs to {output_dir}")


@app.command("stress-test")
def stress_test(
    dataset_dir: Path = typer.Option(Path("data/processed/synthetic_week"), help="Canonical full-horizon dataset directory."),
    n_days: int = typer.Option(4, min=3, help="Initial representative-day count."),
    max_iterations: int = typer.Option(3, min=1, help="Maximum feedback-correction iterations per stress case."),
    output_dir: Path = typer.Option(Path("outputs/stress"), help="Stress-test output directory."),
) -> None:
    """Run predefined robustness stress cases against a scenario."""
    scenario = load_processed_dataset(dataset_dir)
    result = run_stress_suite(scenario, n_typical_days=n_days, max_iterations=max_iterations)
    output_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_dir / "stress_results.csv", index=False, encoding="utf-8-sig")

    table = Table(title="CP202611 Stress Test Summary")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("cases", str(len(result)))
    table.add_row("solved", str(int(result["solved"].sum())))
    table.add_row("all converged", str(bool(result["converged"].fillna(False).all())))
    table.add_row("max comfort slack C", f"{float(result['max_comfort_slack_c'].fillna(0).max()):.3e}")
    table.add_row("max unmet mid MWh", f"{float(result['total_unmet_mid_mwh'].fillna(0).max()):.3e}")
    console.print(table)
    console.print(f"Wrote stress-test outputs to {output_dir}")


@app.command("robustness-matrix")
def robustness_matrix(
    dataset_dir: Path = typer.Option(Path("data/processed/synthetic_season"), help="Canonical processed dataset directory."),
    n_days: int = typer.Option(4, min=3, help="Initial representative-day count."),
    max_iterations: int = typer.Option(3, min=1, help="Maximum feedback-correction iterations per robustness case."),
    output_dir: Path = typer.Option(Path("outputs/robustness_matrix"), help="Robustness-matrix output directory."),
    mode: str = typer.Option("full", help="Robustness suite mode: full or quick."),
) -> None:
    """Compare planning outcomes across exogenous perturbation scenarios."""
    if mode not in {"full", "quick"}:
        raise typer.BadParameter("mode must be one of: full, quick")
    scenario = load_processed_dataset(dataset_dir)
    quick_cases = [
        StressCase(case_id="base"),
        StressCase(case_id="cold_minus_3c", outdoor_delta_c=-3.0),
        StressCase(case_id="grid_limit_minus_25pct", grid_limit_factor=0.75),
    ]
    result = run_robustness_matrix(
        scenario=scenario,
        output_dir=output_dir,
        cases=quick_cases if mode == "quick" else None,
        n_typical_days=n_days,
        max_iterations=max_iterations,
    )

    table = Table(title="CP202611 Robustness Matrix")
    table.add_column("Case")
    table.add_column("Passed", justify="right")
    table.add_column("Converged", justify="right")
    table.add_column("Cost delta %", justify="right")
    table.add_column("Carbon delta %", justify="right")
    table.add_column("Max slack C", justify="right")
    for row in result.matrix.to_dict(orient="records"):
        table.add_row(
            str(row["case_id"]),
            str(row["passed"]),
            str(row["converged"]),
            "" if pd.isna(row.get("cost_delta_pct")) else f"{float(row['cost_delta_pct']):.2f}",
            "" if pd.isna(row.get("carbon_delta_pct")) else f"{float(row['carbon_delta_pct']):.2f}",
            "" if pd.isna(row.get("max_comfort_slack_c")) else f"{float(row['max_comfort_slack_c']):.3e}",
        )
    console.print(table)
    console.print(f"Overall passed: {result.passed}")
    console.print(f"Wrote robustness-matrix outputs to {output_dir}")


@app.command("benchmark-plan")
def benchmark_plan(
    dataset_dir: Path = typer.Option(Path("data/processed/synthetic_season"), help="Canonical full-horizon dataset directory."),
    n_days: int = typer.Option(8, min=3, help="Initial representative-day count."),
    max_iterations: int = typer.Option(4, min=1, help="Maximum feedback-correction iterations."),
    output_dir: Path = typer.Option(Path("outputs/benchmark"), help="Benchmark output directory."),
) -> None:
    """Run an end-to-end planning benchmark and record runtime and residuals."""
    scenario = load_processed_dataset(dataset_dir)
    bundle = run_planning_benchmark_bundle(
        scenario,
        n_typical_days=n_days,
        max_iterations=max_iterations,
    )
    result = bundle.summary
    final = bundle.planning.final_result
    output_dir.mkdir(parents=True, exist_ok=True)
    result.to_dataframe().to_csv(output_dir / "benchmark_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([iteration.__dict__ for iteration in bundle.planning.iterations]).to_csv(
        output_dir / "feedback_iterations.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(
        {
            "day": final.typical_days.selected_days,
            "weight": [final.typical_days.weights[d] for d in final.typical_days.selected_days],
        }
    ).to_csv(output_dir / "final_typical_days.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([final.typical_days.metrics]).to_csv(output_dir / "final_typical_day_metrics.csv", index=False, encoding="utf-8-sig")
    final.planning_result.source_capacity.to_csv(output_dir / "planning_capacity.csv", index=False, encoding="utf-8-sig")
    final.planning_result.spatial_assignment.to_csv(output_dir / "planning_spatial_assignment.csv", index=False, encoding="utf-8-sig")
    final.verification_result.dispatch.to_csv(output_dir / "verification_dispatch.csv", index=False, encoding="utf-8-sig")
    final.verification_result.storage.to_csv(output_dir / "verification_storage.csv", index=False, encoding="utf-8-sig")
    final.verification_result.indoor_temperature.to_csv(output_dir / "verification_indoor_temperature.csv", index=False, encoding="utf-8-sig")
    final.verification_result.network_edges.to_csv(output_dir / "verification_network_edges.csv", index=False, encoding="utf-8-sig")
    final.verification_result.source_capacity.to_csv(output_dir / "verification_capacity.csv", index=False, encoding="utf-8-sig")

    table = Table(title="CP202611 Planning Benchmark")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("scenario", result.scenario_id)
    table.add_row("horizon hours", str(result.horizon_hours))
    table.add_row("selected days", result.selected_days)
    table.add_row("converged", str(result.converged))
    table.add_row("feedback iterations", str(result.feedback_iterations))
    table.add_row("elapsed seconds", f"{result.elapsed_seconds:.2f}")
    table.add_row("comfort slack C", f"{result.max_comfort_slack_c:.3e}")
    table.add_row("unmet mid MWh", f"{result.total_unmet_mid_mwh:.3e}")
    table.add_row("low residual MW", f"{result.max_low_balance_abs_mw:.3e}")
    table.add_row("mid residual MW", f"{result.max_mid_balance_abs_mw:.3e}")
    console.print(table)
    console.print(f"Wrote benchmark outputs to {output_dir}")


@app.command("visualize-results")
def visualize_results(
    result_dir: Path = typer.Option(Path("outputs/feedback_plan"), help="Optimization result directory."),
    dataset_dir: Path = typer.Option(Path("data/processed/synthetic_week"), help="Canonical processed dataset directory."),
    output_dir: Path = typer.Option(Path("outputs/figures"), help="Figure output directory."),
    pareto_dir: Path | None = typer.Option(Path("outputs/pareto"), help="Optional Pareto output directory."),
) -> None:
    """Generate report-ready figures from existing optimization outputs."""
    result = create_report_figures(
        result_dir=result_dir,
        dataset_dir=dataset_dir,
        pareto_dir=pareto_dir,
        output_dir=output_dir,
    )

    table = Table(title="CP202611 Visualization Summary")
    table.add_column("Figure")
    table.add_column("Description")
    for record in result.figures:
        table.add_row(Path(record.file).name, record.description)
    console.print(table)
    console.print(f"Wrote {len(result.figures)} figure files and manifest to {output_dir}")


@app.command("acceptance-check")
def acceptance_check(
    dataset_dir: Path = typer.Option(Path("data/processed/synthetic_season"), help="Canonical processed dataset directory."),
    n_days: int = typer.Option(8, min=3, help="Representative-day count for benchmark planning."),
    max_iterations: int = typer.Option(4, min=1, help="Maximum feedback-correction iterations."),
    carbon_prices: str = typer.Option("0,500,1500", help="Comma-separated carbon prices in CNY/t for acceptance Pareto scan."),
    exergy_penalties: str = typer.Option("0,260", help="Comma-separated exergy penalties in CNY/MWh-eq for acceptance Pareto scan."),
    output_dir: Path = typer.Option(Path("outputs/acceptance"), help="Acceptance-check output directory."),
    stress_mode: str = typer.Option("full", help="Stress suite mode: full or quick."),
    skip_pareto: bool = typer.Option(False, help="Skip Pareto scan for a faster smoke check."),
    skip_visuals: bool = typer.Option(False, help="Skip report figure generation."),
) -> None:
    """Run the pre-submission acceptance workflow on one processed scenario."""
    if stress_mode not in {"full", "quick"}:
        raise typer.BadParameter("stress-mode must be one of: full, quick")
    result = run_acceptance_check(
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        n_typical_days=n_days,
        max_iterations=max_iterations,
        carbon_prices=_parse_float_list(carbon_prices),
        exergy_penalties=_parse_float_list(exergy_penalties),
        stress_cases=[StressCase(case_id="base")] if stress_mode == "quick" else None,
        run_pareto=not skip_pareto,
        run_visuals=not skip_visuals,
    )

    table = Table(title="CP202611 Acceptance Check")
    table.add_column("Check")
    table.add_column("Passed", justify="right")
    table.add_column("Value", justify="right")
    table.add_column("Threshold", justify="right")
    for row in result.summary.to_dict(orient="records"):
        table.add_row(str(row["check"]), str(row["passed"]), str(row["value"]), str(row["threshold"]))
    console.print(table)
    console.print(f"Overall passed: {result.passed}")
    console.print(f"Wrote acceptance outputs to {output_dir}")


@app.command("evidence-package")
def evidence_package(
    output_dir: Path = typer.Option(Path("outputs/evidence_package"), help="Evidence package output directory."),
    acceptance_dir: Path = typer.Option(Path("outputs/acceptance_smoke"), help="Directory containing acceptance_summary.csv."),
    robustness_dir: Path = typer.Option(Path("outputs/robustness_week"), help="Directory containing robustness_matrix.csv."),
    benchmark_dir: Path | None = typer.Option(None, help="Optional directory containing benchmark_summary.csv."),
    dataset_label: str = typer.Option("synthetic_pre_official", help="Dataset label shown in the evidence report."),
) -> None:
    """Build a compact cross-review evidence package from existing outputs."""
    result = build_evidence_package(
        output_dir=output_dir,
        acceptance_dir=acceptance_dir,
        robustness_dir=robustness_dir,
        benchmark_dir=benchmark_dir,
        dataset_label=dataset_label,
    )
    table = Table(title="CP202611 Evidence Package")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("overall ready", str(result.overall_ready))
    table.add_row("acceptance checks", str(result.acceptance_checks))
    table.add_row("robustness cases", str(result.robustness_cases))
    table.add_row("report", str(result.report_path))
    console.print(table)
    console.print(f"Evidence package written to {result.report_path}")


@app.command("typical-days")
def typical_days(
    dataset_dir: Path = typer.Option(Path("data/processed/synthetic_week"), help="Canonical processed dataset directory."),
    n_days: int = typer.Option(4, min=3, help="Number of selected representative days."),
    output: Path = typer.Option(Path("outputs/typical_days.csv"), help="Selected typical-day CSV."),
    metrics_output: Path | None = typer.Option(None, help="Typical-day quality metrics CSV."),
) -> None:
    """Run peak-preserving weighted K-medoids representative-day selection."""
    scenario = load_processed_dataset(dataset_dir)
    time_series = pd.DataFrame(
        {
            "hour": scenario.hours,
            "outdoor_temperature_c": scenario.outdoor_temperature_c,
            "electricity_price_multiplier": scenario.electricity_price_multiplier,
            "grid_carbon_factor_t_per_mwh": scenario.grid_carbon_factor_t_per_mwh,
            "grid_import_limit_mw": scenario.grid_import_limit_mw,
        }
    )
    result = select_peak_preserving_kmedoids(time_series, n_typical_days=n_days)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"day": result.selected_days, "weight": [result.weights[d] for d in result.selected_days]}).to_csv(
        output, index=False, encoding="utf-8-sig"
    )
    metrics_path = metrics_output or output.with_name("typical_day_metrics.csv")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([result.metrics]).to_csv(metrics_path, index=False, encoding="utf-8-sig")
    console.print(f"Selected days: {result.selected_days}")
    console.print(result.metrics)


if __name__ == "__main__":
    app()
