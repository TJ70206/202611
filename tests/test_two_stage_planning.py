from __future__ import annotations

from cp202611.analysis.diagnostics import compute_mvp_diagnostics
from cp202611.planning import (
    build_representative_scenario,
    run_feedback_corrected_planning,
    run_two_stage_planning,
    scenario_time_series,
)
from cp202611.schema import MVPScenario
from cp202611.synthetic import create_synthetic_week
from cp202611.typical_days import select_peak_preserving_kmedoids


def _hidden_mid_temperature_peak_scenario() -> MVPScenario:
    scenario = create_synthetic_week()
    payload = scenario.model_dump()
    payload["max_open_sites"] = 3
    spikes = {
        "residential_a": 0.45,
        "school_b": 0.36,
        "clinic_c": 0.44,
    }
    for building in payload["buildings"]:
        demand = list(building["mid_temp_demand_mw"])
        for hour in range(6 * 24 + 10, 6 * 24 + 15):
            demand[hour] = spikes[building["building_id"]]
        building["mid_temp_demand_mw"] = demand
    return MVPScenario(**payload)


def test_representative_scenario_preserves_weighted_horizon_and_daily_storage_blocks():
    scenario = create_synthetic_week()
    typical_days = select_peak_preserving_kmedoids(scenario_time_series(scenario), n_typical_days=4)

    representative = build_representative_scenario(scenario, typical_days)

    assert len(representative.hours) == 96
    assert representative.storage_cycle_block_size_h == 24
    assert sum(representative.time_weight or []) == len(scenario.hours)
    assert len(representative.buildings[0].mid_temp_demand_mw) == 96


def test_two_stage_planning_verifies_full_week_with_fixed_decisions():
    scenario = create_synthetic_week()

    result = run_two_stage_planning(scenario, n_typical_days=4)
    diagnostics = compute_mvp_diagnostics(result.verification_result)

    assert result.planning_result.termination_condition.lower() == "optimal"
    assert result.verification_result.termination_condition.lower() == "optimal"
    assert diagnostics.max_low_balance_abs_mw < 1e-6
    assert diagnostics.max_mid_balance_abs_mw < 1e-6
    assert diagnostics.max_comfort_slack_c < 1e-5
    assert diagnostics.network_loss_mwh > 0.01


def test_two_stage_verification_uses_planning_capacities_and_topology():
    scenario = create_synthetic_week()

    result = run_two_stage_planning(scenario, n_typical_days=4)
    planned_caps = result.planning_result.source_capacity.set_index("source_id")["capacity_mw"].to_dict()
    verified_caps = result.verification_result.source_capacity.set_index("source_id")["capacity_mw"].to_dict()

    assert planned_caps == verified_caps
    assert result.fixed_decision.storage_capacity_mwh == planned_caps["storage"]
    planned_assignment = result.planning_result.spatial_assignment.sort_values(["building_id", "site_id"]).reset_index(drop=True)
    verified_assignment = result.verification_result.spatial_assignment.sort_values(["building_id", "site_id"]).reset_index(drop=True)
    assert planned_assignment["assigned"].tolist() == verified_assignment["assigned"].tolist()
    assert planned_assignment["site_open"].tolist() == verified_assignment["site_open"].tolist()


def test_feedback_loop_stops_after_one_iteration_when_verification_is_clean():
    scenario = create_synthetic_week()

    result = run_feedback_corrected_planning(scenario, n_typical_days=4, max_iterations=3)

    assert result.converged
    assert len(result.iterations) == 1
    assert result.iterations[0].added_day is None


def test_feedback_loop_adds_hidden_peak_day_and_removes_unmet_heat():
    scenario = _hidden_mid_temperature_peak_scenario()
    first_pass = run_two_stage_planning(scenario, n_typical_days=4)
    first_diagnostics = compute_mvp_diagnostics(first_pass.verification_result)

    result = run_feedback_corrected_planning(scenario, n_typical_days=4, max_iterations=3)
    final_diagnostics = compute_mvp_diagnostics(result.final_result.verification_result)

    assert first_diagnostics.total_unmet_mid_mwh > 0.01
    assert result.converged
    assert any(iteration.added_day == 6 for iteration in result.iterations)
    assert final_diagnostics.total_unmet_mid_mwh < 1e-5
    assert final_diagnostics.max_comfort_slack_c < 1e-5
