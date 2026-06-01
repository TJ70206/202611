from __future__ import annotations

from dataclasses import asdict, dataclass

import pyomo.environ as pyo

from cp202611.analysis.diagnostics import compute_mvp_diagnostics
from cp202611.optimization.mvp_model import SolveResult, cop_for_hour


@dataclass(frozen=True)
class PlanMetrics:
    economic_cost_cny: float
    annualized_capex_cny: float
    fixed_om_cny: float
    operating_cost_cny: float
    variable_om_cny: float
    reliability_penalty_cny: float
    carbon_emissions_t: float
    exergy_loss_mwh_eq: float
    network_loss_mwh: float
    max_comfort_slack_c: float
    total_unmet_mid_mwh: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def evaluate_plan(result: SolveResult) -> PlanMetrics:
    """Compute separable economic, carbon, exergy, and reliability metrics."""
    model = result.model
    data = model._cp202611_data
    sources = {source.source_id: source for source in data.sources}
    sites = {site.site_id: site for site in data.candidate_sites}
    time_weight = data.time_weight or [1.0 for _ in data.hours]

    source_capex = sum(
        pyo.value(model.cap_source[s]) * sources[str(s)].capex_cny_per_mw * data.capex_recovery_factor
        for s in model.S
    )
    source_fixed_om = sum(
        pyo.value(model.cap_source[s]) * sources[str(s)].capex_cny_per_mw * sources[str(s)].fixed_om_fraction
        for s in model.S
    )
    storage_capex = pyo.value(model.cap_storage) * data.storage.capex_cny_per_mwh * data.capex_recovery_factor
    storage_om = pyo.value(model.cap_storage) * data.storage.capex_cny_per_mwh * data.storage.fixed_om_fraction
    site_capex = sum(
        pyo.value(model.site_open[j]) * sites[str(j)].fixed_cost_cny * data.capex_recovery_factor
        for j in model.J
    )

    edge_distance_km = model._cp202611_edge_distance_km
    edge_capacity_mw = model._cp202611_edge_capacity_mw
    pipe_capex = sum(
        pyo.value(model.assign[b, j])
        * edge_distance_km[(str(b), str(j))]
        * edge_capacity_mw[(str(b), str(j))]
        * data.pipe_capex_cny_per_mw_km
        * data.capex_recovery_factor
        for b in model.B
        for j in model.J
    )
    pipe_om = sum(
        pyo.value(model.assign[b, j])
        * edge_distance_km[(str(b), str(j))]
        * edge_capacity_mw[(str(b), str(j))]
        * data.pipe_capex_cny_per_mw_km
        * data.pipe_fixed_om_fraction
        for b in model.B
        for j in model.J
    )

    operating = 0.0
    variable_om = 0.0
    carbon_emissions = 0.0
    exergy_loss = 0.0
    for t in model.T:
        hour = int(t)
        weight = time_weight[hour]
        electricity_price = data.electricity_base_price_cny_per_mwh * data.electricity_price_multiplier[hour]
        for s in model.S:
            source = sources[str(s)]
            q_by_grade = {str(l): pyo.value(model.q[s, l, t]) for l in model.L}
            q_total = sum(q_by_grade.values())
            if source.fuel == "electricity":
                if source.base_cop is not None:
                    fuel_use = q_total / cop_for_hour(source.base_cop, data.outdoor_temperature_c[hour])
                else:
                    fuel_use = q_total / source.efficiency
                operating += weight * electricity_price * fuel_use
                carbon_emissions += weight * data.grid_carbon_factor_t_per_mwh[hour] * fuel_use
            elif source.fuel == "gas":
                fuel_use = q_total / source.efficiency
                operating += weight * data.gas_price_cny_per_mwh_fuel * fuel_use
                carbon_emissions += weight * data.gas_emission_factor_t_per_mwh_fuel * fuel_use
            elif source.fuel == "biomass":
                fuel_use = q_total / source.efficiency
                operating += weight * data.biomass_price_cny_per_mwh_fuel * fuel_use
                carbon_emissions += weight * data.biomass_emission_factor_t_per_mwh_fuel * fuel_use
            else:
                raise ValueError(f"unknown fuel: {source.fuel}")
            variable_om += weight * source.variable_om_cny_per_mwh_th * q_total
            exergy_loss += weight * sum(
                source.exergy_loss_coeff_by_grade[str(grade)] * q_by_grade[str(grade)]
                for grade in model.L
            )

    comfort_penalty = sum(
        data.comfort_slack_penalty_cny_per_c_h
        * pyo.value(model.comfort_low_slack[b, t] + model.comfort_high_slack[b, t])
        * time_weight[min(int(t), len(time_weight) - 1)]
        for b in model.B
        for t in model.TSOC
    )
    unmet_penalty = sum(
        data.unmet_heat_penalty_cny_per_mwh
        * pyo.value(model.unmet_mid[b, t])
        * time_weight[int(t)]
        for b in model.B
        for t in model.T
    )

    annualized_capex = source_capex + storage_capex + site_capex + pipe_capex
    fixed_om = source_fixed_om + storage_om + pipe_om
    reliability_penalty = data.operation_weight * (comfort_penalty + unmet_penalty)
    economic_cost = annualized_capex + fixed_om + data.operation_weight * (operating + variable_om) + reliability_penalty
    diagnostics = compute_mvp_diagnostics(result)
    return PlanMetrics(
        economic_cost_cny=float(economic_cost),
        annualized_capex_cny=float(annualized_capex),
        fixed_om_cny=float(fixed_om),
        operating_cost_cny=float(data.operation_weight * operating),
        variable_om_cny=float(data.operation_weight * variable_om),
        reliability_penalty_cny=float(reliability_penalty),
        carbon_emissions_t=float(carbon_emissions),
        exergy_loss_mwh_eq=float(exergy_loss),
        network_loss_mwh=float(diagnostics.network_loss_mwh),
        max_comfort_slack_c=float(diagnostics.max_comfort_slack_c),
        total_unmet_mid_mwh=float(diagnostics.total_unmet_mid_mwh),
    )
