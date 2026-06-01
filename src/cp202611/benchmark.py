from __future__ import annotations

from dataclasses import asdict, dataclass
from time import perf_counter

import pandas as pd

from cp202611.analysis.diagnostics import compute_mvp_diagnostics
from cp202611.analysis.evaluation import evaluate_plan
from cp202611.planning import FeedbackPlanningResult, run_feedback_corrected_planning
from cp202611.schema import MVPScenario
from cp202611.validation import validate_scenario


@dataclass(frozen=True)
class BenchmarkResult:
    scenario_id: str
    horizon_hours: int
    typical_day_count: int
    selected_days: str
    validation_errors: int
    validation_warnings: int
    converged: bool
    feedback_iterations: int
    elapsed_seconds: float
    objective_value: float
    economic_cost_cny: float
    carbon_emissions_t: float
    exergy_loss_mwh_eq: float
    network_loss_mwh: float
    max_low_balance_abs_mw: float
    max_mid_balance_abs_mw: float
    max_comfort_slack_c: float
    total_unmet_mid_mwh: float

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([asdict(self)])


@dataclass(frozen=True)
class BenchmarkBundle:
    summary: BenchmarkResult
    planning: FeedbackPlanningResult


def run_planning_benchmark(
    scenario: MVPScenario,
    n_typical_days: int = 8,
    max_iterations: int = 4,
) -> BenchmarkResult:
    return run_planning_benchmark_bundle(
        scenario=scenario,
        n_typical_days=n_typical_days,
        max_iterations=max_iterations,
    ).summary


def run_planning_benchmark_bundle(
    scenario: MVPScenario,
    n_typical_days: int = 8,
    max_iterations: int = 4,
) -> BenchmarkBundle:
    validation = validate_scenario(scenario)
    if validation.error_count > 0:
        raise ValueError(f"scenario is not usable: {validation.error_count} validation errors")

    start = perf_counter()
    result = run_feedback_corrected_planning(
        scenario,
        n_typical_days=n_typical_days,
        max_iterations=max_iterations,
    )
    elapsed = perf_counter() - start
    final = result.final_result.verification_result
    diagnostics = compute_mvp_diagnostics(final)
    metrics = evaluate_plan(final)
    summary = BenchmarkResult(
        scenario_id=scenario.scenario_id,
        horizon_hours=len(scenario.hours),
        typical_day_count=len(result.final_result.typical_days.selected_days),
        selected_days=",".join(str(day) for day in result.final_result.typical_days.selected_days),
        validation_errors=validation.error_count,
        validation_warnings=validation.warning_count,
        converged=result.converged,
        feedback_iterations=len(result.iterations),
        elapsed_seconds=float(elapsed),
        objective_value=final.objective_value,
        economic_cost_cny=metrics.economic_cost_cny,
        carbon_emissions_t=metrics.carbon_emissions_t,
        exergy_loss_mwh_eq=metrics.exergy_loss_mwh_eq,
        network_loss_mwh=metrics.network_loss_mwh,
        max_low_balance_abs_mw=diagnostics.max_low_balance_abs_mw,
        max_mid_balance_abs_mw=diagnostics.max_mid_balance_abs_mw,
        max_comfort_slack_c=diagnostics.max_comfort_slack_c,
        total_unmet_mid_mwh=diagnostics.total_unmet_mid_mwh,
    )
    return BenchmarkBundle(summary=summary, planning=result)
