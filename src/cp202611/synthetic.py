from __future__ import annotations

import math

from cp202611.schema import (
    Building,
    CandidateSite,
    HeatSource,
    MVPScenario,
    StorageSpec,
)


def _tou_multiplier(hour: int) -> float:
    hour_of_day = hour % 24
    if 2 <= hour_of_day <= 5 or 11 <= hour_of_day <= 14:
        return 0.30
    if 16 <= hour_of_day <= 20:
        return 1.60
    return 1.00


def _outdoor_temp(hour: int) -> float:
    day = hour // 24
    cold_spell = -4.0 if 3 <= day <= 4 else 0.0
    return -5.0 + cold_spell + 5.0 * math.sin((hour - 7) / 24.0 * 2.0 * math.pi)


def _season_outdoor_temp(hour: int, n_hours: int) -> float:
    day = hour // 24
    n_days = max(1, n_hours // 24)
    day_ratio = day / max(1, n_days - 1)
    seasonal_base = -2.0 - 5.0 * math.sin(math.pi * day_ratio)
    diurnal = 4.0 * math.sin((hour - 7) / 24.0 * 2.0 * math.pi)
    cold_spell = -3.0 if 24 <= day <= 26 or 64 <= day <= 66 else 0.0
    return seasonal_base + diurnal + cold_spell


def _create_synthetic_scenario(n_hours: int, outdoor: list[float], scenario_id: str) -> MVPScenario:
    hours = list(range(n_hours))
    multipliers = [_tou_multiplier(h) for h in hours]
    grid_limit = [1.05 if 16 <= h % 24 <= 20 else 1.8 for h in hours]

    buildings = [
        Building(
            building_id="residential_a",
            lon=117.02,
            lat=36.66,
            floor_area_m2=18000,
            heat_loss_mw_per_c=0.035,
            thermal_capacity_mwh_per_c=0.42,
            initial_indoor_temp_c=19.0,
            comfort_min_c=18.0,
            comfort_max_c=22.0,
            internal_gain_mw=0.035,
            mid_temp_demand_mw=[0.045 + (0.015 if 6 <= h % 24 <= 8 or 18 <= h % 24 <= 21 else 0) for h in hours],
            peak_heat_mw=1.15,
        ),
        Building(
            building_id="school_b",
            lon=117.035,
            lat=36.665,
            floor_area_m2=12000,
            heat_loss_mw_per_c=0.025,
            thermal_capacity_mwh_per_c=0.30,
            initial_indoor_temp_c=19.0,
            comfort_min_c=18.0,
            comfort_max_c=22.0,
            internal_gain_mw=0.020,
            mid_temp_demand_mw=[0.025 if 7 <= h % 24 <= 17 else 0.010 for h in hours],
            peak_heat_mw=0.82,
        ),
        Building(
            building_id="clinic_c",
            lon=117.055,
            lat=36.650,
            floor_area_m2=9000,
            heat_loss_mw_per_c=0.022,
            thermal_capacity_mwh_per_c=0.25,
            initial_indoor_temp_c=19.5,
            comfort_min_c=18.5,
            comfort_max_c=22.5,
            internal_gain_mw=0.018,
            mid_temp_demand_mw=[0.040 + (0.012 if 8 <= h % 24 <= 20 else 0) for h in hours],
            peak_heat_mw=0.75,
        ),
    ]

    sources = [
        HeatSource(
            source_id="ashp_low",
            fuel="electricity",
            allowed_grades=["low"],
            max_capacity_mw=1.60,
            capex_cny_per_mw=3_500_000,
            fixed_om_fraction=0.020,
            efficiency=1.0,
            base_cop=2.70,
            variable_om_cny_per_mwh_th=10.0,
            exergy_loss_coeff_by_grade={"low": 0.05, "mid": 0.05},
        ),
        HeatSource(
            source_id="gas_boiler",
            fuel="gas",
            allowed_grades=["low", "mid"],
            max_capacity_mw=1.60,
            capex_cny_per_mw=850_000,
            fixed_om_fraction=0.015,
            efficiency=0.90,
            variable_om_cny_per_mwh_th=8.0,
            exergy_loss_coeff_by_grade={"low": 0.28, "mid": 0.10},
        ),
        HeatSource(
            source_id="biomass_boiler",
            fuel="biomass",
            allowed_grades=["low", "mid"],
            max_capacity_mw=0.25,
            capex_cny_per_mw=1_200_000,
            fixed_om_fraction=0.030,
            efficiency=0.82,
            variable_om_cny_per_mwh_th=18.0,
            exergy_loss_coeff_by_grade={"low": 0.18, "mid": 0.08},
        ),
        HeatSource(
            source_id="electric_boiler",
            fuel="electricity",
            allowed_grades=["low", "mid"],
            max_capacity_mw=0.22,
            capex_cny_per_mw=200_000,
            fixed_om_fraction=0.010,
            efficiency=0.98,
            variable_om_cny_per_mwh_th=5.0,
            exergy_loss_coeff_by_grade={"low": 0.32, "mid": 0.16},
        ),
    ]

    sites = [
        CandidateSite(site_id="site_west", lon=117.015, lat=36.660, max_radius_km=3.0, service_capacity_mw=1.60, fixed_cost_cny=80_000),
        CandidateSite(site_id="site_central", lon=117.038, lat=36.660, max_radius_km=3.0, service_capacity_mw=2.00, fixed_cost_cny=110_000),
        CandidateSite(site_id="site_east", lon=117.060, lat=36.650, max_radius_km=3.0, service_capacity_mw=1.40, fixed_cost_cny=75_000),
    ]

    return MVPScenario(
        scenario_id=scenario_id,
        hours=hours,
        outdoor_temperature_c=outdoor,
        electricity_price_multiplier=multipliers,
        grid_carbon_factor_t_per_mwh=[0.5777 for _ in hours],
        grid_import_limit_mw=grid_limit,
        electricity_base_price_cny_per_mwh=800.0,
        gas_price_cny_per_mwh_fuel=422.0,
        biomass_price_cny_per_mwh_fuel=192.0,
        gas_emission_factor_t_per_mwh_fuel=0.202,
        biomass_emission_factor_t_per_mwh_fuel=0.035,
        carbon_price_cny_per_t=80.0,
        capex_recovery_factor=0.10,
        operation_weight=120.0,
        exergy_penalty_cny_per_mwh=260.0,
        comfort_slack_penalty_cny_per_c_h=18_000.0,
        unmet_heat_penalty_cny_per_mwh=60_000.0,
        max_open_sites=2,
        pipe_loss_fraction_per_km=0.015,
        pipe_capex_cny_per_mw_km=2_500_000,
        pipe_fixed_om_fraction=0.015,
        pipe_capacity_margin=1.25,
        sources=sources,
        buildings=buildings,
        candidate_sites=sites,
        storage=StorageSpec(
            max_capacity_mwh=2.20,
            capex_cny_per_mwh=200_000,
            fixed_om_fraction=0.008,
            charge_efficiency=0.96,
            discharge_efficiency=0.96,
            standing_loss_fraction_per_h=0.0005,
            power_to_energy_ratio=0.65,
        ),
    )


def create_synthetic_mvp(n_hours: int = 24) -> MVPScenario:
    outdoor = [_outdoor_temp(h) for h in range(n_hours)]
    return _create_synthetic_scenario(
        n_hours=n_hours,
        outdoor=outdoor,
        scenario_id="synthetic_north_small_city_mvp",
    )


def create_synthetic_week() -> MVPScenario:
    return create_synthetic_mvp(n_hours=168)


def create_synthetic_season(n_hours: int = 2880) -> MVPScenario:
    if n_hours < 24 or n_hours % 24 != 0:
        raise ValueError("synthetic season requires a positive whole-day horizon")
    outdoor = [_season_outdoor_temp(h, n_hours) for h in range(n_hours)]
    return _create_synthetic_scenario(
        n_hours=n_hours,
        outdoor=outdoor,
        scenario_id="synthetic_north_small_city_season",
    )
