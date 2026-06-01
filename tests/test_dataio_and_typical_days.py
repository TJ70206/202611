from __future__ import annotations

import pandas as pd
import pytest

from cp202611.dataio import load_processed_dataset, write_processed_dataset
from cp202611.schema import MVPScenario
from cp202611.optimization.mvp_model import solve_mvp
from cp202611.synthetic import create_synthetic_week
from cp202611.typical_days import select_peak_preserving_kmedoids


def test_processed_dataset_roundtrip_preserves_core_counts(tmp_path):
    scenario = create_synthetic_week()
    payload = scenario.model_dump()
    payload["route_distance_factor"] = 1.45
    payload["indoor_target_penalty_cny_per_c_h"] = 150.0
    scenario = MVPScenario(**payload)

    write_processed_dataset(scenario, tmp_path)

    loaded = load_processed_dataset(tmp_path)

    assert loaded.scenario_id == scenario.scenario_id
    assert len(loaded.hours) == 168
    assert len(loaded.sources) == len(scenario.sources)
    assert len(loaded.buildings) == len(scenario.buildings)
    assert loaded.buildings[0].mid_temp_demand_mw == scenario.buildings[0].mid_temp_demand_mw
    assert loaded.pipe_loss_fraction_per_km == scenario.pipe_loss_fraction_per_km
    assert loaded.pipe_capex_cny_per_mw_km == scenario.pipe_capex_cny_per_mw_km
    assert loaded.route_distance_factor == pytest.approx(1.45)
    assert loaded.indoor_target_penalty_cny_per_c_h == pytest.approx(150.0)


def test_week_scenario_solves_after_csv_yaml_roundtrip(tmp_path):
    scenario = create_synthetic_week()
    write_processed_dataset(scenario, tmp_path)
    loaded = load_processed_dataset(tmp_path)

    result = solve_mvp(loaded)

    assert result.termination_condition.lower() == "optimal"
    assert len(result.storage) == 169


def test_synthetic_peak_heat_covers_extreme_steady_state_load():
    scenario = create_synthetic_week()
    min_outdoor = min(scenario.outdoor_temperature_c)

    for building in scenario.buildings:
        required = building.heat_loss_mw_per_c * (building.comfort_min_c - min_outdoor) - building.internal_gain_mw
        assert building.peak_heat_mw >= required


def test_typical_days_force_extreme_conditions():
    scenario = create_synthetic_week()
    time_series = pd.DataFrame(
        {
            "hour": scenario.hours,
            "outdoor_temperature_c": scenario.outdoor_temperature_c,
            "electricity_price_multiplier": scenario.electricity_price_multiplier,
            "grid_carbon_factor_t_per_mwh": scenario.grid_carbon_factor_t_per_mwh,
            "grid_import_limit_mw": scenario.grid_import_limit_mw,
        }
    )

    result = select_peak_preserving_kmedoids(time_series, n_typical_days=4)

    assert len(result.selected_days) == 4
    assert result.metrics["selected_min_temperature_c"] == result.metrics["full_min_temperature_c"]
    assert result.metrics["selected_max_price_multiplier"] == result.metrics["full_max_price_multiplier"]
    assert sum(result.weights.values()) == 7
    assert all(weight >= 1 for weight in result.weights.values())


def test_typical_days_requires_enough_extreme_slots():
    scenario = create_synthetic_week()
    time_series = pd.DataFrame(
        {
            "hour": scenario.hours,
            "outdoor_temperature_c": scenario.outdoor_temperature_c,
            "electricity_price_multiplier": scenario.electricity_price_multiplier,
            "grid_carbon_factor_t_per_mwh": scenario.grid_carbon_factor_t_per_mwh,
            "grid_import_limit_mw": scenario.grid_import_limit_mw,
        }
    )

    with pytest.raises(ValueError, match="at least 3"):
        select_peak_preserving_kmedoids(time_series, n_typical_days=2)


def test_scenario_requires_zero_based_consecutive_hours():
    scenario = create_synthetic_week()
    payload = scenario.model_dump()
    payload["hours"] = list(range(1, len(scenario.hours) + 1))

    with pytest.raises(ValueError, match="zero-based consecutive"):
        MVPScenario(**payload)
