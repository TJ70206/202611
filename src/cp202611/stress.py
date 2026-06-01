from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd

from cp202611.analysis.evaluation import evaluate_plan
from cp202611.planning import run_feedback_corrected_planning
from cp202611.schema import MVPScenario
from cp202611.validation import validate_scenario


@dataclass(frozen=True)
class StressCase:
    case_id: str
    outdoor_delta_c: float = 0.0
    grid_limit_factor: float = 1.0
    ashp_cop_factor: float = 1.0
    pipe_loss_factor: float = 1.0
    gas_price_factor: float = 1.0
    biomass_price_factor: float = 1.0


DEFAULT_STRESS_CASES = [
    StressCase(case_id="base"),
    StressCase(case_id="cold_minus_5c", outdoor_delta_c=-5.0),
    StressCase(case_id="ashp_cop_minus_15pct", ashp_cop_factor=0.85),
    StressCase(case_id="grid_limit_minus_25pct", grid_limit_factor=0.75),
    StressCase(case_id="grid_limit_minus_55pct", grid_limit_factor=0.45),
    StressCase(case_id="pipe_loss_plus_50pct", pipe_loss_factor=1.50),
    StressCase(case_id="fuel_price_plus_30pct", gas_price_factor=1.30, biomass_price_factor=1.30),
]


def apply_stress_case(scenario: MVPScenario, case: StressCase) -> MVPScenario:
    payload = scenario.model_dump()
    payload["scenario_id"] = f"{scenario.scenario_id}_{case.case_id}"
    payload["outdoor_temperature_c"] = [value + case.outdoor_delta_c for value in scenario.outdoor_temperature_c]
    payload["grid_import_limit_mw"] = [value * case.grid_limit_factor for value in scenario.grid_import_limit_mw]
    payload["pipe_loss_fraction_per_km"] = scenario.pipe_loss_fraction_per_km * case.pipe_loss_factor
    payload["gas_price_cny_per_mwh_fuel"] = scenario.gas_price_cny_per_mwh_fuel * case.gas_price_factor
    payload["biomass_price_cny_per_mwh_fuel"] = scenario.biomass_price_cny_per_mwh_fuel * case.biomass_price_factor
    for source in payload["sources"]:
        if str(source["source_id"]).startswith("ashp") and source.get("base_cop") is not None:
            source["base_cop"] = float(source["base_cop"]) * case.ashp_cop_factor
    min_outdoor = min(payload["outdoor_temperature_c"])
    for building in payload["buildings"]:
        steady_peak = (
            float(building["heat_loss_mw_per_c"]) * (float(building["comfort_min_c"]) - min_outdoor)
            - float(building.get("internal_gain_mw", 0.0))
        )
        building["peak_heat_mw"] = max(float(building["peak_heat_mw"]), steady_peak)
    return MVPScenario(**payload)


def run_stress_suite(
    scenario: MVPScenario,
    cases: list[StressCase] | None = None,
    n_typical_days: int = 4,
    max_iterations: int = 3,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for case in cases or DEFAULT_STRESS_CASES:
        stressed = apply_stress_case(scenario, case)
        validation = validate_scenario(stressed)
        row: dict[str, object] = asdict(case)
        row.update(
            {
                "validation_errors": validation.error_count,
                "validation_warnings": validation.warning_count,
            }
        )
        if validation.error_count > 0:
            row.update(
                {
                    "solved": False,
                    "converged": False,
                    "feedback_iterations": 0,
                    "economic_cost_cny": None,
                    "carbon_emissions_t": None,
                    "exergy_loss_mwh_eq": None,
                    "max_comfort_slack_c": None,
                    "total_unmet_mid_mwh": None,
                    "network_loss_mwh": None,
                }
            )
            rows.append(row)
            continue

        result = run_feedback_corrected_planning(
            stressed,
            n_typical_days=n_typical_days,
            max_iterations=max_iterations,
        )
        metrics = evaluate_plan(result.final_result.verification_result)
        row.update(
            {
                "solved": True,
                "converged": result.converged,
                "feedback_iterations": len(result.iterations),
                "economic_cost_cny": metrics.economic_cost_cny,
                "carbon_emissions_t": metrics.carbon_emissions_t,
                "exergy_loss_mwh_eq": metrics.exergy_loss_mwh_eq,
                "max_comfort_slack_c": metrics.max_comfort_slack_c,
                "total_unmet_mid_mwh": metrics.total_unmet_mid_mwh,
                "network_loss_mwh": metrics.network_loss_mwh,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)
