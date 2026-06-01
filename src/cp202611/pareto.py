from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from cp202611.analysis.evaluation import evaluate_plan
from cp202611.planning import run_feedback_corrected_planning
from cp202611.schema import MVPScenario


@dataclass(frozen=True)
class ParetoScanResult:
    runs: pd.DataFrame
    pareto_front: pd.DataFrame


def with_objective_weights(
    scenario: MVPScenario,
    carbon_price_cny_per_t: float,
    exergy_penalty_cny_per_mwh: float,
) -> MVPScenario:
    payload = scenario.model_dump()
    payload["carbon_price_cny_per_t"] = carbon_price_cny_per_t
    payload["exergy_penalty_cny_per_mwh"] = exergy_penalty_cny_per_mwh
    return MVPScenario(**payload)


def run_pareto_scan(
    scenario: MVPScenario,
    carbon_prices: list[float],
    exergy_penalties: list[float],
    n_typical_days: int = 4,
    max_iterations: int = 4,
) -> ParetoScanResult:
    rows: list[dict[str, float | int | bool]] = []
    run_id = 0
    for carbon_price in carbon_prices:
        for exergy_penalty in exergy_penalties:
            run_id += 1
            weighted_scenario = with_objective_weights(
                scenario,
                carbon_price_cny_per_t=carbon_price,
                exergy_penalty_cny_per_mwh=exergy_penalty,
            )
            result = run_feedback_corrected_planning(
                weighted_scenario,
                n_typical_days=n_typical_days,
                max_iterations=max_iterations,
            )
            metrics = evaluate_plan(result.final_result.verification_result)
            capacities = result.final_result.verification_result.source_capacity.set_index("source_id")[
                "capacity_mw"
            ].to_dict()
            rows.append(
                {
                    "run_id": run_id,
                    "carbon_price_cny_per_t": float(carbon_price),
                    "exergy_penalty_cny_per_mwh": float(exergy_penalty),
                    "converged": bool(result.converged),
                    "feedback_iterations": len(result.iterations),
                    "typical_day_count": len(result.final_result.typical_days.selected_days),
                    "economic_cost_cny": metrics.economic_cost_cny,
                    "carbon_emissions_t": metrics.carbon_emissions_t,
                    "exergy_loss_mwh_eq": metrics.exergy_loss_mwh_eq,
                    "network_loss_mwh": metrics.network_loss_mwh,
                    "max_comfort_slack_c": metrics.max_comfort_slack_c,
                    "total_unmet_mid_mwh": metrics.total_unmet_mid_mwh,
                    "ashp_capacity_mw": float(capacities.get("ashp_low", 0.0)),
                    "gas_capacity_mw": float(capacities.get("gas_boiler", 0.0)),
                    "biomass_capacity_mw": float(capacities.get("biomass_boiler", 0.0)),
                    "electric_boiler_capacity_mw": float(capacities.get("electric_boiler", 0.0)),
                    "storage_capacity_mwh": float(capacities.get("storage", 0.0)),
                }
            )
    runs = pd.DataFrame(rows)
    pareto_front = mark_pareto_front(
        runs,
        objective_cols=["economic_cost_cny", "carbon_emissions_t", "exergy_loss_mwh_eq"],
    )
    return ParetoScanResult(runs=runs, pareto_front=pareto_front)


def mark_pareto_front(runs: pd.DataFrame, objective_cols: list[str]) -> pd.DataFrame:
    if runs.empty:
        return runs.copy()

    values = runs[objective_cols].to_numpy(dtype=float)
    is_pareto = []
    for i, point in enumerate(values):
        dominated = False
        for j, other in enumerate(values):
            if i == j:
                continue
            if (other <= point).all() and (other < point).any():
                dominated = True
                break
        is_pareto.append(not dominated)
    front = runs.copy()
    front["is_pareto"] = is_pareto
    return front[front["is_pareto"]].sort_values(objective_cols).reset_index(drop=True)
