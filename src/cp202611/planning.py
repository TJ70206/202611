from __future__ import annotations

from dataclasses import dataclass

import pyomo.environ as pyo
import pandas as pd

from cp202611.analysis.diagnostics import compute_mvp_diagnostics
from cp202611.optimization.mvp_model import FixedPlanningDecision, SolveResult, extract_fixed_decision, solve_mvp
from cp202611.schema import MVPScenario
from cp202611.typical_days import TypicalDayResult, rebuild_typical_day_result, select_peak_preserving_kmedoids


@dataclass(frozen=True)
class TwoStageResult:
    typical_days: TypicalDayResult
    planning_scenario: MVPScenario
    planning_result: SolveResult
    fixed_decision: FixedPlanningDecision
    verification_result: SolveResult


@dataclass(frozen=True)
class FeedbackIteration:
    iteration: int
    selected_days: list[int]
    added_day: int | None
    worst_day: int | None
    max_comfort_slack_c: float
    total_unmet_mid_mwh: float
    planning_objective: float
    verification_objective: float


@dataclass(frozen=True)
class FeedbackPlanningResult:
    final_result: TwoStageResult
    iterations: list[FeedbackIteration]
    converged: bool


def scenario_time_series(scenario: MVPScenario) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "hour": scenario.hours,
            "outdoor_temperature_c": scenario.outdoor_temperature_c,
            "electricity_price_multiplier": scenario.electricity_price_multiplier,
            "grid_carbon_factor_t_per_mwh": scenario.grid_carbon_factor_t_per_mwh,
            "grid_import_limit_mw": scenario.grid_import_limit_mw,
        }
    )


def build_representative_scenario(full_scenario: MVPScenario, typical_days: TypicalDayResult) -> MVPScenario:
    """Build a weighted representative-day planning scenario from a full scenario."""
    old_hours: list[int] = []
    time_weight: list[float] = []
    for day in typical_days.selected_days:
        day_start = int(day) * 24
        day_hours = list(range(day_start, day_start + 24))
        if day_hours[-1] >= len(full_scenario.hours):
            raise ValueError(f"selected day {day} is outside the scenario horizon")
        old_hours.extend(day_hours)
        time_weight.extend([float(typical_days.weights[int(day)]) for _ in day_hours])

    buildings = []
    for building in full_scenario.buildings:
        buildings.append(
            building.model_copy(
                deep=True,
                update={"mid_temp_demand_mw": [building.mid_temp_demand_mw[h] for h in old_hours]},
            )
        )

    payload = full_scenario.model_dump()
    payload.update(
        {
            "scenario_id": f"{full_scenario.scenario_id}_typical_days",
            "hours": list(range(len(old_hours))),
            "outdoor_temperature_c": [full_scenario.outdoor_temperature_c[h] for h in old_hours],
            "electricity_price_multiplier": [full_scenario.electricity_price_multiplier[h] for h in old_hours],
            "grid_carbon_factor_t_per_mwh": [full_scenario.grid_carbon_factor_t_per_mwh[h] for h in old_hours],
            "grid_import_limit_mw": [full_scenario.grid_import_limit_mw[h] for h in old_hours],
            "time_weight": time_weight,
            "storage_cycle_block_size_h": 24,
            "buildings": [building.model_dump() for building in buildings],
        }
    )
    return MVPScenario(**payload)


def run_two_stage_planning(full_scenario: MVPScenario, n_typical_days: int = 4) -> TwoStageResult:
    """Run weighted representative-day sizing, then full-horizon verification with fixed decisions."""
    typical_days = select_peak_preserving_kmedoids(
        scenario_time_series(full_scenario),
        n_typical_days=n_typical_days,
    )
    planning_scenario = build_representative_scenario(full_scenario, typical_days)
    planning_result = solve_mvp(planning_scenario)
    fixed_decision = extract_fixed_decision(planning_result)
    verification_result = solve_mvp(full_scenario, fixed_decision=fixed_decision)
    return TwoStageResult(
        typical_days=typical_days,
        planning_scenario=planning_scenario,
        planning_result=planning_result,
        fixed_decision=fixed_decision,
        verification_result=verification_result,
    )


def run_feedback_corrected_planning(
    full_scenario: MVPScenario,
    n_typical_days: int = 4,
    max_iterations: int = 4,
    comfort_tolerance_c: float = 1e-5,
    unmet_tolerance_mwh: float = 1e-5,
) -> FeedbackPlanningResult:
    """Iteratively add worst verification days to representative-day planning."""
    if max_iterations < 1:
        raise ValueError("max_iterations must be at least 1")

    time_series = scenario_time_series(full_scenario)
    typical_days = select_peak_preserving_kmedoids(time_series, n_typical_days=n_typical_days)
    iterations: list[FeedbackIteration] = []
    final_result: TwoStageResult | None = None
    converged = False

    for iteration in range(1, max_iterations + 1):
        planning_scenario = build_representative_scenario(full_scenario, typical_days)
        planning_result = solve_mvp(planning_scenario)
        fixed_decision = extract_fixed_decision(planning_result)
        verification_result = solve_mvp(full_scenario, fixed_decision=fixed_decision)
        final_result = TwoStageResult(
            typical_days=typical_days,
            planning_scenario=planning_scenario,
            planning_result=planning_result,
            fixed_decision=fixed_decision,
            verification_result=verification_result,
        )

        diagnostics = compute_mvp_diagnostics(verification_result)
        worst_day = _worst_violation_day(verification_result)
        needs_feedback = (
            diagnostics.max_comfort_slack_c > comfort_tolerance_c
            or diagnostics.total_unmet_mid_mwh > unmet_tolerance_mwh
        )
        added_day: int | None = None
        if needs_feedback and worst_day is not None and worst_day not in typical_days.selected_days:
            added_day = worst_day

        iterations.append(
            FeedbackIteration(
                iteration=iteration,
                selected_days=list(typical_days.selected_days),
                added_day=added_day,
                worst_day=worst_day if needs_feedback else None,
                max_comfort_slack_c=diagnostics.max_comfort_slack_c,
                total_unmet_mid_mwh=diagnostics.total_unmet_mid_mwh,
                planning_objective=planning_result.objective_value,
                verification_objective=verification_result.objective_value,
            )
        )

        if not needs_feedback:
            converged = True
            break
        if added_day is None:
            break
        typical_days = rebuild_typical_day_result(time_series, typical_days.selected_days + [added_day])

    if final_result is None:
        raise RuntimeError("feedback planning did not run")
    return FeedbackPlanningResult(final_result=final_result, iterations=iterations, converged=converged)


def _worst_violation_day(result: SolveResult) -> int | None:
    """Return the day with the largest unmet heat or comfort violation."""
    model = result.model
    data = model._cp202611_data
    rows: list[dict[str, float]] = []
    for t in model.T:
        hour = int(t)
        comfort_slack = float(
            result.indoor_temperature[result.indoor_temperature["hour"] == hour][
                ["low_slack_c", "high_slack_c"]
            ].sum(axis=1).sum()
        )
        unmet_mid = sum(float(pyo.value(model.unmet_mid[b, t])) for b in model.B)
        rows.append({"hour": hour, "day": hour // 24, "comfort_slack_c": comfort_slack, "unmet_mid_mwh": unmet_mid})
    if not rows:
        return None
    frame = pd.DataFrame(rows)
    by_day = frame.groupby("day", as_index=False)[["comfort_slack_c", "unmet_mid_mwh"]].sum()
    by_day["score"] = by_day["unmet_mid_mwh"] + by_day["comfort_slack_c"]
    worst = by_day.sort_values("score", ascending=False).iloc[0]
    if float(worst["score"]) <= 0:
        return None
    day = int(worst["day"])
    max_day = (len(data.hours) - 1) // 24
    return min(day, max_day)
