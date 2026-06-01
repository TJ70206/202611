from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from cp202611.analysis.diagnostics import compute_mvp_diagnostics
from cp202611.analysis.evaluation import evaluate_plan
from cp202611.planning import run_feedback_corrected_planning
from cp202611.schema import MVPScenario
from cp202611.stress import DEFAULT_STRESS_CASES, StressCase, apply_stress_case
from cp202611.validation import validate_scenario


@dataclass(frozen=True)
class RobustnessCriteria:
    max_comfort_slack_c: float = 1e-5
    max_unmet_mid_mwh: float = 1e-5
    max_balance_residual_mw: float = 1e-6


@dataclass(frozen=True)
class RobustnessMatrixResult:
    passed: bool
    matrix: pd.DataFrame


def run_robustness_matrix(
    scenario: MVPScenario,
    output_dir: Path,
    cases: list[StressCase] | None = None,
    n_typical_days: int = 4,
    max_iterations: int = 3,
    criteria: RobustnessCriteria | None = None,
) -> RobustnessMatrixResult:
    """Run planning on multiple exogenous perturbations and report pass/fail deltas."""
    criteria = criteria or RobustnessCriteria()
    case_list = cases or DEFAULT_STRESS_CASES
    if not any(case.case_id == "base" for case in case_list):
        raise ValueError("robustness matrix requires a 'base' case for delta calculations")
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = [
        _run_case(
            scenario=scenario,
            case=case,
            n_typical_days=n_typical_days,
            max_iterations=max_iterations,
            criteria=criteria,
        )
        for case in case_list
    ]
    matrix = pd.DataFrame(rows)
    matrix = _add_baseline_deltas(matrix)
    passed = bool(matrix["passed"].fillna(False).all())

    matrix.to_csv(output_dir / "robustness_matrix.csv", index=False, encoding="utf-8-sig")
    (output_dir / "robustness_report.md").write_text(
        _matrix_to_markdown(matrix, passed, criteria),
        encoding="utf-8",
    )
    return RobustnessMatrixResult(passed=passed, matrix=matrix)


def _run_case(
    scenario: MVPScenario,
    case: StressCase,
    n_typical_days: int,
    max_iterations: int,
    criteria: RobustnessCriteria,
) -> dict[str, object]:
    stressed = apply_stress_case(scenario, case)
    validation = validate_scenario(stressed)
    row: dict[str, object] = {
        "source_scenario_id": scenario.scenario_id,
        **asdict(case),
        "validation_errors": validation.error_count,
        "validation_warnings": validation.warning_count,
    }
    if validation.error_count > 0:
        row.update(_failed_metrics(error_message=f"{validation.error_count} validation errors"))
        return row

    try:
        planning = run_feedback_corrected_planning(
            stressed,
            n_typical_days=n_typical_days,
            max_iterations=max_iterations,
        )
    except Exception as exc:  # pragma: no cover - defensive report path for external data.
        row.update(_failed_metrics(error_message=f"{type(exc).__name__}: {exc}"))
        return row

    final = planning.final_result.verification_result
    diagnostics = compute_mvp_diagnostics(final)
    metrics = evaluate_plan(final)
    passed = (
        planning.converged
        and diagnostics.max_low_balance_abs_mw <= criteria.max_balance_residual_mw
        and diagnostics.max_mid_balance_abs_mw <= criteria.max_balance_residual_mw
        and diagnostics.max_comfort_slack_c <= criteria.max_comfort_slack_c
        and diagnostics.total_unmet_mid_mwh <= criteria.max_unmet_mid_mwh
    )
    row.update(
        {
            "solved": True,
            "converged": planning.converged,
            "passed": bool(passed),
            "feedback_iterations": len(planning.iterations),
            "typical_days": ",".join(str(day) for day in planning.final_result.typical_days.selected_days),
            "economic_cost_cny": metrics.economic_cost_cny,
            "carbon_emissions_t": metrics.carbon_emissions_t,
            "exergy_loss_mwh_eq": metrics.exergy_loss_mwh_eq,
            "network_loss_mwh": metrics.network_loss_mwh,
            "max_low_balance_abs_mw": diagnostics.max_low_balance_abs_mw,
            "max_mid_balance_abs_mw": diagnostics.max_mid_balance_abs_mw,
            "max_comfort_slack_c": diagnostics.max_comfort_slack_c,
            "total_unmet_mid_mwh": diagnostics.total_unmet_mid_mwh,
            "error_message": "",
        }
    )
    return row


def _failed_metrics(error_message: str) -> dict[str, object]:
    return {
        "solved": False,
        "converged": False,
        "passed": False,
        "feedback_iterations": 0,
        "typical_days": "",
        "economic_cost_cny": None,
        "carbon_emissions_t": None,
        "exergy_loss_mwh_eq": None,
        "network_loss_mwh": None,
        "max_low_balance_abs_mw": None,
        "max_mid_balance_abs_mw": None,
        "max_comfort_slack_c": None,
        "total_unmet_mid_mwh": None,
        "error_message": error_message,
    }


def _add_baseline_deltas(matrix: pd.DataFrame) -> pd.DataFrame:
    matrix = matrix.copy()
    baseline_rows = matrix.loc[matrix["case_id"] == "base"]
    if baseline_rows.empty:
        raise ValueError("robustness matrix requires a 'base' case for delta calculations")
    baseline = baseline_rows.iloc[0]
    delta_specs = {
        "economic_cost_cny": "cost_delta_pct",
        "carbon_emissions_t": "carbon_delta_pct",
        "exergy_loss_mwh_eq": "exergy_delta_pct",
        "network_loss_mwh": "network_loss_delta_pct",
    }
    for metric, column in delta_specs.items():
        base_value = baseline.get(metric)
        if pd.isna(base_value) or float(base_value) == 0:
            matrix[column] = None
        else:
            matrix[column] = (matrix[metric] - float(base_value)) / float(base_value) * 100.0
    return matrix


def _matrix_to_markdown(matrix: pd.DataFrame, passed: bool, criteria: RobustnessCriteria) -> str:
    lines = [
        "# CP202611 Robustness Matrix Report",
        "",
        f"- overall_passed: `{passed}`",
        f"- cases: `{len(matrix)}`",
        f"- comfort_slack_limit_c: `{criteria.max_comfort_slack_c}`",
        f"- unmet_mid_limit_mwh: `{criteria.max_unmet_mid_mwh}`",
        "",
        "| case | passed | converged | cost_delta_% | carbon_delta_% | exergy_delta_% | comfort_slack_C | unmet_mid_MWh |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in matrix.to_dict(orient="records"):
        lines.append(
            "| {case_id} | {passed} | {converged} | {cost} | {carbon} | {exergy} | {slack} | {unmet} |".format(
                case_id=row["case_id"],
                passed=row["passed"],
                converged=row["converged"],
                cost=_fmt(row.get("cost_delta_pct")),
                carbon=_fmt(row.get("carbon_delta_pct")),
                exergy=_fmt(row.get("exergy_delta_pct")),
                slack=_fmt(row.get("max_comfort_slack_c"), scientific=True),
                unmet=_fmt(row.get("total_unmet_mid_mwh"), scientific=True),
            )
        )
    failed = matrix.loc[~matrix["passed"].fillna(False)]
    if not failed.empty:
        lines.extend(["", "## Failed Cases", ""])
        for row in failed.to_dict(orient="records"):
            message = row.get("error_message") or "criteria not satisfied"
            lines.append(f"- `{row['case_id']}`: {message}")
    return "\n".join(lines) + "\n"


def _fmt(value: object, scientific: bool = False) -> str:
    if value is None or pd.isna(value):
        return ""
    number = float(value)
    if scientific:
        return f"{number:.3e}"
    return f"{number:.2f}"
