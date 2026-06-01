from __future__ import annotations

from cp202611.schema import MVPScenario
from cp202611.stress import StressCase, apply_stress_case, run_stress_suite
from cp202611.synthetic import create_synthetic_week
from cp202611.validation import validate_scenario


def test_validate_synthetic_scenario_has_no_errors():
    scenario = create_synthetic_week()

    report = validate_scenario(scenario)

    assert report.is_usable
    assert report.error_count == 0


def test_validate_scenario_flags_peak_heat_below_extreme_load():
    scenario = create_synthetic_week()
    payload = scenario.model_dump()
    payload["buildings"][0]["peak_heat_mw"] = 0.01
    broken = MVPScenario(**payload)

    report = validate_scenario(broken)

    assert not report.is_usable
    assert any(issue.code == "PEAK_HEAT_BELOW_STEADY_EXTREME" for issue in report.issues)


def test_apply_stress_case_changes_expected_fields():
    scenario = create_synthetic_week()
    case = StressCase(case_id="probe", outdoor_delta_c=-3.0, grid_limit_factor=0.8, pipe_loss_factor=1.5)

    stressed = apply_stress_case(scenario, case)

    assert stressed.scenario_id.endswith("_probe")
    assert min(stressed.outdoor_temperature_c) == min(scenario.outdoor_temperature_c) - 3.0
    assert stressed.grid_import_limit_mw[0] == scenario.grid_import_limit_mw[0] * 0.8
    assert stressed.pipe_loss_fraction_per_km == scenario.pipe_loss_fraction_per_km * 1.5
    assert stressed.buildings[0].peak_heat_mw >= scenario.buildings[0].peak_heat_mw


def test_stress_suite_runs_clean_cases_without_unmet_heat():
    scenario = create_synthetic_week()
    cases = [
        StressCase(case_id="base"),
        StressCase(case_id="cold_minus_3c", outdoor_delta_c=-3.0),
        StressCase(case_id="pipe_loss_plus_10pct", pipe_loss_factor=1.10),
    ]

    result = run_stress_suite(scenario, cases=cases, n_typical_days=4, max_iterations=2)

    assert len(result) == 3
    assert result["solved"].all()
    assert result["converged"].all()
    assert (result["total_unmet_mid_mwh"] < 1e-5).all()
