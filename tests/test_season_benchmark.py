from __future__ import annotations

import pytest

from cp202611.benchmark import run_planning_benchmark
from cp202611.dataio import load_processed_dataset, write_processed_dataset
from cp202611.synthetic import create_synthetic_season
from cp202611.validation import validate_scenario


def test_synthetic_season_has_whole_day_horizon_and_passes_validation():
    scenario = create_synthetic_season(n_hours=240)
    report = validate_scenario(scenario)

    assert len(scenario.hours) == 240
    assert scenario.scenario_id == "synthetic_north_small_city_season"
    assert min(scenario.outdoor_temperature_c) < max(scenario.outdoor_temperature_c)
    assert report.error_count == 0


def test_synthetic_season_roundtrip_preserves_profile(tmp_path):
    scenario = create_synthetic_season(n_hours=240)

    write_processed_dataset(scenario, tmp_path)
    loaded = load_processed_dataset(tmp_path)

    assert loaded.scenario_id == scenario.scenario_id
    assert loaded.outdoor_temperature_c == pytest.approx(scenario.outdoor_temperature_c)
    assert loaded.buildings[0].peak_heat_mw == scenario.buildings[0].peak_heat_mw


def test_planning_benchmark_runs_on_short_season():
    scenario = create_synthetic_season(n_hours=240)

    result = run_planning_benchmark(scenario, n_typical_days=4, max_iterations=2)

    assert result.horizon_hours == 240
    assert result.converged
    assert result.elapsed_seconds > 0
    assert result.max_comfort_slack_c < 1e-5
    assert result.total_unmet_mid_mwh < 1e-5
