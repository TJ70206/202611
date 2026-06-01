from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

TemperatureGrade = Literal["low", "mid"]
FuelType = Literal["electricity", "gas", "biomass"]


class HeatSource(BaseModel):
    source_id: str
    fuel: FuelType
    allowed_grades: list[TemperatureGrade]
    max_capacity_mw: float = Field(gt=0)
    capex_cny_per_mw: float = Field(ge=0)
    fixed_om_fraction: float = Field(ge=0, le=1)
    efficiency: float = Field(gt=0)
    base_cop: float | None = None
    variable_om_cny_per_mwh_th: float = Field(default=0, ge=0)
    exergy_loss_coeff_by_grade: dict[TemperatureGrade, float]

    @field_validator("allowed_grades")
    @classmethod
    def grades_not_empty(cls, value: list[TemperatureGrade]) -> list[TemperatureGrade]:
        if not value:
            raise ValueError("allowed_grades must not be empty")
        return value

    @model_validator(mode="after")
    def check_cop_for_electric_heat_pump(self) -> "HeatSource":
        if self.source_id.startswith("ashp") and self.base_cop is None:
            raise ValueError("ASHP source requires base_cop")
        return self


class Building(BaseModel):
    building_id: str
    lon: float
    lat: float
    floor_area_m2: float = Field(gt=0)
    heat_loss_mw_per_c: float = Field(gt=0)
    thermal_capacity_mwh_per_c: float = Field(gt=0)
    initial_indoor_temp_c: float
    comfort_min_c: float
    comfort_max_c: float
    internal_gain_mw: float = Field(default=0, ge=0)
    mid_temp_demand_mw: list[float]
    peak_heat_mw: float = Field(gt=0)

    @model_validator(mode="after")
    def comfort_band_is_valid(self) -> "Building":
        if self.comfort_min_c >= self.comfort_max_c:
            raise ValueError("comfort_min_c must be lower than comfort_max_c")
        return self


class CandidateSite(BaseModel):
    site_id: str
    lon: float
    lat: float
    max_radius_km: float = Field(gt=0)
    service_capacity_mw: float = Field(gt=0)
    fixed_cost_cny: float = Field(default=0, ge=0)


class StorageSpec(BaseModel):
    storage_id: str = "tank_low"
    max_capacity_mwh: float = Field(gt=0)
    capex_cny_per_mwh: float = Field(ge=0)
    fixed_om_fraction: float = Field(ge=0, le=1)
    charge_efficiency: float = Field(gt=0, le=1)
    discharge_efficiency: float = Field(gt=0, le=1)
    standing_loss_fraction_per_h: float = Field(ge=0, lt=1)
    power_to_energy_ratio: float = Field(gt=0)


class MVPScenario(BaseModel):
    scenario_id: str
    hours: list[int]
    outdoor_temperature_c: list[float]
    electricity_price_multiplier: list[float]
    grid_carbon_factor_t_per_mwh: list[float]
    grid_import_limit_mw: list[float]
    time_weight: list[float] | None = None
    electricity_base_price_cny_per_mwh: float = Field(gt=0)
    gas_price_cny_per_mwh_fuel: float = Field(gt=0)
    biomass_price_cny_per_mwh_fuel: float = Field(gt=0)
    gas_emission_factor_t_per_mwh_fuel: float = Field(ge=0)
    biomass_emission_factor_t_per_mwh_fuel: float = Field(ge=0)
    carbon_price_cny_per_t: float = Field(ge=0)
    capex_recovery_factor: float = Field(gt=0, le=1)
    operation_weight: float = Field(gt=0)
    exergy_penalty_cny_per_mwh: float = Field(ge=0)
    comfort_slack_penalty_cny_per_c_h: float = Field(gt=0)
    unmet_heat_penalty_cny_per_mwh: float = Field(gt=0)
    max_open_sites: int = Field(gt=0)
    pipe_loss_fraction_per_km: float = Field(default=0.015, ge=0, le=0.2)
    pipe_capex_cny_per_mw_km: float = Field(default=2_500_000, ge=0)
    pipe_fixed_om_fraction: float = Field(default=0.015, ge=0, le=1)
    pipe_capacity_margin: float = Field(default=1.25, gt=0)
    storage_cycle_block_size_h: int | None = Field(default=None, gt=0)
    sources: list[HeatSource]
    buildings: list[Building]
    candidate_sites: list[CandidateSite]
    storage: StorageSpec

    @model_validator(mode="after")
    def time_series_lengths_match(self) -> "MVPScenario":
        n = len(self.hours)
        if self.hours != list(range(n)):
            raise ValueError("hours must be a zero-based consecutive index: 0..n-1")
        series = {
            "outdoor_temperature_c": self.outdoor_temperature_c,
            "electricity_price_multiplier": self.electricity_price_multiplier,
            "grid_carbon_factor_t_per_mwh": self.grid_carbon_factor_t_per_mwh,
            "grid_import_limit_mw": self.grid_import_limit_mw,
        }
        if self.time_weight is not None:
            series["time_weight"] = self.time_weight
        for name, values in series.items():
            if len(values) != n:
                raise ValueError(f"{name} length must equal hours length")
        if self.time_weight is not None and any(weight <= 0 for weight in self.time_weight):
            raise ValueError("time_weight values must be positive")
        if self.storage_cycle_block_size_h is not None and n % self.storage_cycle_block_size_h != 0:
            raise ValueError("hours length must be divisible by storage_cycle_block_size_h")
        for building in self.buildings:
            if len(building.mid_temp_demand_mw) != n:
                raise ValueError(f"{building.building_id} mid_temp_demand_mw length mismatch")
        return self
