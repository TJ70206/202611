from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import pandas as pd

from cp202611.optimization.mvp_model import distance_km
from cp202611.schema import MVPScenario

Severity = Literal["error", "warning", "info"]


@dataclass(frozen=True)
class DataIssue:
    severity: Severity
    code: str
    message: str
    object_id: str = ""
    value: float | str | None = None
    threshold: float | str | None = None


@dataclass(frozen=True)
class ValidationReport:
    scenario_id: str
    issues: list[DataIssue]

    @property
    def error_count(self) -> int:
        return sum(issue.severity == "error" for issue in self.issues)

    @property
    def warning_count(self) -> int:
        return sum(issue.severity == "warning" for issue in self.issues)

    @property
    def is_usable(self) -> bool:
        return self.error_count == 0

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([asdict(issue) for issue in self.issues])

    def to_markdown(self) -> str:
        lines = [
            f"# Validation Report: {self.scenario_id}",
            "",
            f"- usable: `{self.is_usable}`",
            f"- errors: `{self.error_count}`",
            f"- warnings: `{self.warning_count}`",
            "",
        ]
        if not self.issues:
            lines.append("No issues found.")
            return "\n".join(lines)
        lines.append("| severity | code | object_id | value | threshold | message |")
        lines.append("|---|---|---|---:|---:|---|")
        for issue in self.issues:
            lines.append(
                f"| {issue.severity} | {issue.code} | {issue.object_id} | "
                f"{'' if issue.value is None else issue.value} | "
                f"{'' if issue.threshold is None else issue.threshold} | {issue.message} |"
            )
        return "\n".join(lines)


def validate_scenario(scenario: MVPScenario) -> ValidationReport:
    issues: list[DataIssue] = []
    _check_time_series(scenario, issues)
    _check_buildings(scenario, issues)
    _check_sources(scenario, issues)
    _check_spatial_feasibility(scenario, issues)
    _check_parameter_ranges(scenario, issues)
    if not issues:
        issues.append(DataIssue("info", "NO_ISSUES", "Scenario passed all implemented validation checks."))
    return ValidationReport(scenario_id=scenario.scenario_id, issues=issues)


def _add(issues: list[DataIssue], severity: Severity, code: str, message: str, object_id: str = "", value=None, threshold=None) -> None:
    issues.append(DataIssue(severity, code, message, object_id, value, threshold))


def _check_time_series(scenario: MVPScenario, issues: list[DataIssue]) -> None:
    n = len(scenario.hours)
    if n < 24:
        _add(issues, "warning", "SHORT_HORIZON", "Scenario horizon is shorter than 24 hours.", value=n, threshold=24)
    if min(scenario.outdoor_temperature_c) < -45 or max(scenario.outdoor_temperature_c) > 45:
        _add(
            issues,
            "warning",
            "OUTDOOR_TEMPERATURE_RANGE",
            "Outdoor temperature range is unusual; check units and source city.",
            value=f"{min(scenario.outdoor_temperature_c):.2f}..{max(scenario.outdoor_temperature_c):.2f}",
            threshold="-45..45 C",
        )
    for name, values in {
        "electricity_price_multiplier": scenario.electricity_price_multiplier,
        "grid_import_limit_mw": scenario.grid_import_limit_mw,
    }.items():
        if any(value <= 0 for value in values):
            _add(issues, "error", "NON_POSITIVE_TIME_SERIES", f"{name} contains non-positive values.", object_id=name)
    if any(value < 0 for value in scenario.grid_carbon_factor_t_per_mwh):
        _add(issues, "error", "NEGATIVE_CARBON_FACTOR", "Grid carbon factor contains negative values.", object_id="grid_carbon_factor_t_per_mwh")


def _check_buildings(scenario: MVPScenario, issues: list[DataIssue]) -> None:
    min_outdoor = min(scenario.outdoor_temperature_c)
    for building in scenario.buildings:
        if any(value < 0 for value in building.mid_temp_demand_mw):
            _add(issues, "error", "NEGATIVE_MID_DEMAND", "Mid-temperature demand contains negative values.", building.building_id)
        steady_peak = building.heat_loss_mw_per_c * (building.comfort_min_c - min_outdoor) - building.internal_gain_mw
        if building.peak_heat_mw + 1e-9 < steady_peak:
            _add(
                issues,
                "error",
                "PEAK_HEAT_BELOW_STEADY_EXTREME",
                "Building peak_heat_mw is lower than the steady load required at the coldest outdoor temperature.",
                building.building_id,
                round(building.peak_heat_mw, 4),
                round(steady_peak, 4),
            )
        heat_loss_intensity = building.heat_loss_mw_per_c * 1_000_000.0 / building.floor_area_m2
        if heat_loss_intensity < 0.1 or heat_loss_intensity > 10:
            _add(
                issues,
                "warning",
                "HEAT_LOSS_INTENSITY_RANGE",
                "Building heat-loss intensity is outside a broad engineering sanity range.",
                building.building_id,
                round(heat_loss_intensity, 3),
                "0.1..10 W/(m2.K)",
            )


def _check_sources(scenario: MVPScenario, issues: list[DataIssue]) -> None:
    for grade in ["low", "mid"]:
        grade_capacity = sum(
            source.max_capacity_mw for source in scenario.sources if grade in source.allowed_grades
        )
        if grade_capacity <= 0:
            _add(issues, "error", "NO_SOURCE_FOR_GRADE", f"No heat source can serve grade {grade}.", grade)

    low_capacity = sum(source.max_capacity_mw for source in scenario.sources if "low" in source.allowed_grades)
    mid_capacity = sum(source.max_capacity_mw for source in scenario.sources if "mid" in source.allowed_grades)
    min_outdoor = min(scenario.outdoor_temperature_c)
    low_peak = sum(
        max(0.0, building.heat_loss_mw_per_c * (building.comfort_min_c - min_outdoor) - building.internal_gain_mw)
        for building in scenario.buildings
    )
    mid_peak = max(
        sum(building.mid_temp_demand_mw[t] for building in scenario.buildings)
        for t in range(len(scenario.hours))
    )
    if low_capacity < low_peak:
        _add(issues, "warning", "LOW_SOURCE_CAPACITY_MARGIN", "Low-temperature source capacity is below extreme steady load.", value=round(low_capacity, 3), threshold=round(low_peak, 3))
    if mid_capacity < mid_peak:
        _add(issues, "error", "MID_SOURCE_CAPACITY_MARGIN", "Mid-temperature source capacity is below peak mid-temperature demand.", value=round(mid_capacity, 3), threshold=round(mid_peak, 3))


def _check_spatial_feasibility(scenario: MVPScenario, issues: list[DataIssue]) -> None:
    for building in scenario.buildings:
        reachable = [
            site.site_id
            for site in scenario.candidate_sites
            if distance_km(building.lon, building.lat, site.lon, site.lat) <= site.max_radius_km
        ]
        if not reachable:
            _add(issues, "error", "BUILDING_WITHOUT_REACHABLE_SITE", "Building has no candidate site within max service radius.", building.building_id)

    required_capacity = sum(building.peak_heat_mw + max(building.mid_temp_demand_mw) for building in scenario.buildings)
    available_capacity = sum(
        site.service_capacity_mw
        for site in sorted(scenario.candidate_sites, key=lambda site: site.service_capacity_mw, reverse=True)[: scenario.max_open_sites]
    )
    if available_capacity + 1e-9 < required_capacity:
        _add(
            issues,
            "error",
            "SITE_SERVICE_CAPACITY_INSUFFICIENT",
            "Top candidate sites allowed by max_open_sites cannot cover aggregate building peak service capacity.",
            value=round(available_capacity, 3),
            threshold=round(required_capacity, 3),
        )


def _check_parameter_ranges(scenario: MVPScenario, issues: list[DataIssue]) -> None:
    if scenario.pipe_loss_fraction_per_km > 0.05:
        _add(issues, "warning", "PIPE_LOSS_HIGH", "Pipe loss fraction per km is high for a planning model.", value=scenario.pipe_loss_fraction_per_km, threshold=0.05)
    if scenario.gas_price_cny_per_mwh_fuel < 200 or scenario.gas_price_cny_per_mwh_fuel > 800:
        _add(issues, "warning", "GAS_PRICE_RANGE", "Gas price is outside a broad Chinese non-residential sanity range after MWh conversion.", value=scenario.gas_price_cny_per_mwh_fuel, threshold="200..800")
    if scenario.biomass_price_cny_per_mwh_fuel < 80 or scenario.biomass_price_cny_per_mwh_fuel > 350:
        _add(issues, "warning", "BIOMASS_PRICE_RANGE", "Biomass fuel price is outside a broad sanity range after MWh conversion.", value=scenario.biomass_price_cny_per_mwh_fuel, threshold="80..350")
