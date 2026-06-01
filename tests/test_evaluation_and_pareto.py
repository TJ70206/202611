from __future__ import annotations

import pandas as pd

from cp202611.analysis.evaluation import evaluate_plan
from cp202611.pareto import mark_pareto_front, run_pareto_scan
from cp202611.planning import run_feedback_corrected_planning
from cp202611.synthetic import create_synthetic_week


def test_evaluate_plan_reports_separate_positive_metrics():
    scenario = create_synthetic_week()
    result = run_feedback_corrected_planning(scenario, n_typical_days=4, max_iterations=2)

    metrics = evaluate_plan(result.final_result.verification_result)

    assert metrics.economic_cost_cny > 0
    assert metrics.carbon_emissions_t > 0
    assert metrics.exergy_loss_mwh_eq > 0
    assert metrics.network_loss_mwh > 0
    assert metrics.reliability_penalty_cny == 0


def test_mark_pareto_front_removes_dominated_rows():
    runs = pd.DataFrame(
        [
            {"run_id": 1, "economic_cost_cny": 10.0, "carbon_emissions_t": 10.0, "exergy_loss_mwh_eq": 10.0},
            {"run_id": 2, "economic_cost_cny": 9.0, "carbon_emissions_t": 9.0, "exergy_loss_mwh_eq": 9.0},
            {"run_id": 3, "economic_cost_cny": 8.0, "carbon_emissions_t": 12.0, "exergy_loss_mwh_eq": 8.0},
        ]
    )

    front = mark_pareto_front(runs, ["economic_cost_cny", "carbon_emissions_t", "exergy_loss_mwh_eq"])

    assert set(front["run_id"]) == {2, 3}


def test_pareto_scan_runs_feedback_planning_and_returns_front():
    scenario = create_synthetic_week()

    result = run_pareto_scan(
        scenario,
        carbon_prices=[0.0, 160.0],
        exergy_penalties=[0.0],
        n_typical_days=4,
        max_iterations=2,
    )

    assert len(result.runs) == 2
    assert not result.pareto_front.empty
    assert result.runs["converged"].all()
    assert (result.runs["total_unmet_mid_mwh"] < 1e-5).all()
    assert {"economic_cost_cny", "carbon_emissions_t", "exergy_loss_mwh_eq"}.issubset(result.runs.columns)
