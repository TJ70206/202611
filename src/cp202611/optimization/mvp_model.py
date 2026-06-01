from __future__ import annotations

from dataclasses import dataclass
from math import cos, radians, sqrt
from typing import Any

import pandas as pd
import pyomo.environ as pyo

from cp202611.schema import MVPScenario


@dataclass(frozen=True)
class SolveResult:
    model: pyo.ConcreteModel
    termination_condition: str
    objective_value: float
    dispatch: pd.DataFrame
    indoor_temperature: pd.DataFrame
    storage: pd.DataFrame
    spatial_assignment: pd.DataFrame
    network_edges: pd.DataFrame
    source_capacity: pd.DataFrame


@dataclass(frozen=True)
class FixedPlanningDecision:
    source_capacity_mw: dict[str, float]
    storage_capacity_mwh: float
    site_open: dict[str, int]
    assignment: dict[tuple[str, str], int]


def distance_km(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    mean_lat = radians((lat1 + lat2) / 2.0)
    dx = (lon1 - lon2) * 111.320 * cos(mean_lat)
    dy = (lat1 - lat2) * 110.574
    return sqrt(dx * dx + dy * dy)


def cop_for_hour(base_cop: float, outdoor_c: float) -> float:
    # Conservative linear proxy: colder outdoor air lowers ASHP COP.
    return max(1.60, min(3.40, base_cop + 0.045 * (outdoor_c - 0.0)))


def build_mvp_model(data: MVPScenario, fixed_decision: FixedPlanningDecision | None = None) -> pyo.ConcreteModel:
    m = pyo.ConcreteModel(name=data.scenario_id)

    hours = data.hours
    end_hour = len(hours)
    source_ids = [s.source_id for s in data.sources]
    building_ids = [b.building_id for b in data.buildings]
    site_ids = [s.site_id for s in data.candidate_sites]
    grades = ["low", "mid"]

    sources = {s.source_id: s for s in data.sources}
    buildings = {b.building_id: b for b in data.buildings}
    sites = {s.site_id: s for s in data.candidate_sites}
    time_weight = data.time_weight or [1.0 for _ in hours]

    m.T = pyo.Set(initialize=hours, ordered=True)
    m.TSOC = pyo.Set(initialize=list(range(end_hour + 1)), ordered=True)
    m.S = pyo.Set(initialize=source_ids)
    m.B = pyo.Set(initialize=building_ids)
    m.J = pyo.Set(initialize=site_ids)
    m.L = pyo.Set(initialize=grades)

    m.cap_source = pyo.Var(m.S, domain=pyo.NonNegativeReals)
    m.q = pyo.Var(m.S, m.L, m.T, domain=pyo.NonNegativeReals)
    m.q_room = pyo.Var(m.B, m.T, domain=pyo.NonNegativeReals)
    m.unmet_mid = pyo.Var(m.B, m.T, domain=pyo.NonNegativeReals)
    m.tin = pyo.Var(m.B, m.TSOC, domain=pyo.Reals)
    m.comfort_low_slack = pyo.Var(m.B, m.TSOC, domain=pyo.NonNegativeReals)
    m.comfort_high_slack = pyo.Var(m.B, m.TSOC, domain=pyo.NonNegativeReals)
    m.charge = pyo.Var(m.T, domain=pyo.NonNegativeReals)
    m.discharge = pyo.Var(m.T, domain=pyo.NonNegativeReals)
    m.soc = pyo.Var(m.TSOC, domain=pyo.NonNegativeReals)
    m.cap_storage = pyo.Var(domain=pyo.NonNegativeReals, bounds=(0, data.storage.max_capacity_mwh))
    m.site_open = pyo.Var(m.J, domain=pyo.Binary)
    m.assign = pyo.Var(m.B, m.J, domain=pyo.Binary)
    m.edge_low = pyo.Var(m.B, m.J, m.T, domain=pyo.NonNegativeReals)
    m.edge_mid = pyo.Var(m.B, m.J, m.T, domain=pyo.NonNegativeReals)

    for s_id, source in sources.items():
        m.cap_source[s_id].setub(source.max_capacity_mw)

    if fixed_decision is not None:
        for s_id in source_ids:
            m.cap_source[s_id].fix(fixed_decision.source_capacity_mw[s_id])
        m.cap_storage.fix(fixed_decision.storage_capacity_mwh)
        for j_id in site_ids:
            m.site_open[j_id].fix(fixed_decision.site_open[j_id])
        for b_id in building_ids:
            for j_id in site_ids:
                m.assign[b_id, j_id].fix(fixed_decision.assignment[(b_id, j_id)])

    def source_capacity_rule(model: pyo.ConcreteModel, s: str, t: int) -> pyo.Expression:
        return sum(model.q[s, l, t] for l in model.L) <= model.cap_source[s]

    m.source_capacity_limit = pyo.Constraint(m.S, m.T, rule=source_capacity_rule)

    def grade_capability_rule(model: pyo.ConcreteModel, s: str, l: str, t: int) -> pyo.Expression:
        if l not in sources[s].allowed_grades:
            return model.q[s, l, t] == 0
        return pyo.Constraint.Skip

    m.grade_capability = pyo.Constraint(m.S, m.L, m.T, rule=grade_capability_rule)

    def storage_balance_rule(model: pyo.ConcreteModel, t: int) -> pyo.Expression:
        st = data.storage
        return model.soc[t + 1] == (
            model.soc[t] * (1.0 - st.standing_loss_fraction_per_h)
            + st.charge_efficiency * model.charge[t]
            - model.discharge[t] / st.discharge_efficiency
        )

    m.storage_balance = pyo.Constraint(m.T, rule=storage_balance_rule)
    if data.storage_cycle_block_size_h is None:
        m.storage_cycle = pyo.Constraint(expr=m.soc[end_hour] == m.soc[0])
    else:
        block_size = data.storage_cycle_block_size_h
        block_starts = list(range(0, end_hour, block_size))
        m.StorageCycleBlocks = pyo.Set(initialize=block_starts, ordered=True)
        m.storage_cycle = pyo.Constraint(
            m.StorageCycleBlocks,
            rule=lambda model, start: model.soc[start + block_size] == model.soc[start],
        )
    m.soc_capacity = pyo.Constraint(m.TSOC, rule=lambda model, t: model.soc[t] <= model.cap_storage)
    m.charge_power = pyo.Constraint(m.T, rule=lambda model, t: model.charge[t] <= data.storage.power_to_energy_ratio * model.cap_storage)
    m.discharge_power = pyo.Constraint(m.T, rule=lambda model, t: model.discharge[t] <= data.storage.power_to_energy_ratio * model.cap_storage)

    def initial_temp_rule(model: pyo.ConcreteModel, b: str) -> pyo.Expression:
        return model.tin[b, 0] == buildings[b].initial_indoor_temp_c

    m.initial_temp = pyo.Constraint(m.B, rule=initial_temp_rule)

    def rc_rule(model: pyo.ConcreteModel, b: str, t: int) -> pyo.Expression:
        building = buildings[b]
        delta = (
            model.q_room[b, t]
            + building.internal_gain_mw
            - building.heat_loss_mw_per_c * (model.tin[b, t] - data.outdoor_temperature_c[t])
        ) / building.thermal_capacity_mwh_per_c
        return model.tin[b, t + 1] == model.tin[b, t] + delta

    m.rc_balance = pyo.Constraint(m.B, m.T, rule=rc_rule)

    m.comfort_min = pyo.Constraint(
        m.B,
        m.TSOC,
        rule=lambda model, b, t: model.tin[b, t] + model.comfort_low_slack[b, t] >= buildings[b].comfort_min_c,
    )
    m.comfort_max = pyo.Constraint(
        m.B,
        m.TSOC,
        rule=lambda model, b, t: model.tin[b, t] - model.comfort_high_slack[b, t] <= buildings[b].comfort_max_c,
    )

    m.one_site_per_building = pyo.Constraint(m.B, rule=lambda model, b: sum(model.assign[b, j] for j in model.J) == 1)
    m.assign_only_open_site = pyo.Constraint(m.B, m.J, rule=lambda model, b, j: model.assign[b, j] <= model.site_open[j])
    m.max_open_sites = pyo.Constraint(expr=sum(m.site_open[j] for j in m.J) <= data.max_open_sites)

    allowed_pairs: dict[tuple[str, str], int] = {}
    edge_distance_km: dict[tuple[str, str], float] = {}
    edge_loss_fraction: dict[tuple[str, str], float] = {}
    edge_capacity_mw: dict[tuple[str, str], float] = {}
    for b_id, building in buildings.items():
        for j_id, site in sites.items():
            distance = distance_km(building.lon, building.lat, site.lon, site.lat)
            edge_distance_km[(b_id, j_id)] = distance
            edge_loss_fraction[(b_id, j_id)] = data.pipe_loss_fraction_per_km * distance
            edge_capacity_mw[(b_id, j_id)] = data.pipe_capacity_margin * (
                building.peak_heat_mw + max(building.mid_temp_demand_mw)
            )
            allowed_pairs[(b_id, j_id)] = int(
                distance <= site.max_radius_km
            )

    m.spatial_radius = pyo.Constraint(
        m.B,
        m.J,
        rule=lambda model, b, j: model.assign[b, j] <= allowed_pairs[(b, j)],
    )
    m.site_service_capacity = pyo.Constraint(
        m.J,
        rule=lambda model, j: sum(
            (buildings[b].peak_heat_mw + max(buildings[b].mid_temp_demand_mw)) * model.assign[b, j]
            for b in model.B
        )
        <= sites[j].service_capacity_mw * model.site_open[j],
    )

    m.low_edge_delivery = pyo.Constraint(
        m.B,
        m.T,
        rule=lambda model, b, t: sum(model.edge_low[b, j, t] for j in model.J) == model.q_room[b, t],
    )
    m.mid_edge_delivery = pyo.Constraint(
        m.B,
        m.T,
        rule=lambda model, b, t: sum(model.edge_mid[b, j, t] for j in model.J) + model.unmet_mid[b, t]
        == buildings[b].mid_temp_demand_mw[t],
    )
    m.edge_low_assignment_limit = pyo.Constraint(
        m.B,
        m.J,
        m.T,
        rule=lambda model, b, j, t: model.edge_low[b, j, t]
        <= max(buildings[b].peak_heat_mw * 2.0, 1.0) * model.assign[b, j],
    )
    m.edge_mid_assignment_limit = pyo.Constraint(
        m.B,
        m.J,
        m.T,
        rule=lambda model, b, j, t: model.edge_mid[b, j, t]
        <= max(max(buildings[b].mid_temp_demand_mw) * 2.0, 0.05) * model.assign[b, j],
    )
    m.edge_capacity = pyo.Constraint(
        m.B,
        m.J,
        m.T,
        rule=lambda model, b, j, t: (model.edge_low[b, j, t] + model.edge_mid[b, j, t])
        * (1.0 + edge_loss_fraction[(b, j)])
        <= edge_capacity_mw[(b, j)] * model.assign[b, j],
    )

    def low_heat_balance_rule(model: pyo.ConcreteModel, t: int) -> pyo.Expression:
        supply = sum(model.q[s, "low", t] for s in model.S) + model.discharge[t]
        pipe_use = sum(
            model.edge_low[b, j, t] * (1.0 + edge_loss_fraction[(b, j)])
            for b in model.B
            for j in model.J
        )
        return supply == pipe_use + model.charge[t]

    m.low_heat_balance = pyo.Constraint(m.T, rule=low_heat_balance_rule)

    def mid_heat_balance_rule(model: pyo.ConcreteModel, t: int) -> pyo.Expression:
        supply = sum(model.q[s, "mid", t] for s in model.S)
        pipe_use = sum(
            model.edge_mid[b, j, t] * (1.0 + edge_loss_fraction[(b, j)])
            for b in model.B
            for j in model.J
        )
        return supply == pipe_use

    m.mid_heat_balance = pyo.Constraint(m.T, rule=mid_heat_balance_rule)

    def electricity_use(model: pyo.ConcreteModel, t: int) -> pyo.Expression:
        total = 0
        for s_id in model.S:
            source = sources[s_id]
            q_total = sum(model.q[s_id, l, t] for l in model.L)
            if source.fuel == "electricity":
                if source.base_cop is not None:
                    total += q_total / cop_for_hour(source.base_cop, data.outdoor_temperature_c[t])
                else:
                    total += q_total / source.efficiency
        return total

    m.grid_limit = pyo.Constraint(m.T, rule=lambda model, t: electricity_use(model, t) <= data.grid_import_limit_mw[t])

    def source_fuel_use(model: pyo.ConcreteModel, s_id: str, t: int) -> pyo.Expression:
        source = sources[s_id]
        q_total = sum(model.q[s_id, l, t] for l in model.L)
        if source.fuel == "electricity":
            if source.base_cop is not None:
                return q_total / cop_for_hour(source.base_cop, data.outdoor_temperature_c[t])
            return q_total / source.efficiency
        return q_total / source.efficiency

    def objective_rule(model: pyo.ConcreteModel) -> pyo.Expression:
        source_capex = sum(
            model.cap_source[s] * sources[s].capex_cny_per_mw * data.capex_recovery_factor
            for s in model.S
        )
        source_fixed_om = sum(
            model.cap_source[s] * sources[s].capex_cny_per_mw * sources[s].fixed_om_fraction
            for s in model.S
        )
        storage_capex = model.cap_storage * data.storage.capex_cny_per_mwh * data.capex_recovery_factor
        storage_om = model.cap_storage * data.storage.capex_cny_per_mwh * data.storage.fixed_om_fraction
        site_cost = sum(model.site_open[j] * sites[j].fixed_cost_cny * data.capex_recovery_factor for j in model.J)
        pipe_capex = sum(
            model.assign[b, j]
            * edge_distance_km[(b, j)]
            * edge_capacity_mw[(b, j)]
            * data.pipe_capex_cny_per_mw_km
            * data.capex_recovery_factor
            for b in model.B
            for j in model.J
        )
        pipe_fixed_om = sum(
            model.assign[b, j]
            * edge_distance_km[(b, j)]
            * edge_capacity_mw[(b, j)]
            * data.pipe_capex_cny_per_mw_km
            * data.pipe_fixed_om_fraction
            for b in model.B
            for j in model.J
        )

        operating = 0
        carbon = 0
        exergy = 0
        source_variable_om = 0
        for t in model.T:
            elec_price = data.electricity_base_price_cny_per_mwh * data.electricity_price_multiplier[t]
            weight = time_weight[int(t)]
            for s in model.S:
                source = sources[s]
                q_total = sum(model.q[s, l, t] for l in model.L)
                fuel_use = source_fuel_use(model, s, t)
                if source.fuel == "electricity":
                    operating += weight * elec_price * fuel_use
                    carbon += weight * data.grid_carbon_factor_t_per_mwh[t] * fuel_use * data.carbon_price_cny_per_t
                elif source.fuel == "gas":
                    operating += weight * data.gas_price_cny_per_mwh_fuel * fuel_use
                    carbon += weight * data.gas_emission_factor_t_per_mwh_fuel * fuel_use * data.carbon_price_cny_per_t
                elif source.fuel == "biomass":
                    operating += weight * data.biomass_price_cny_per_mwh_fuel * fuel_use
                    carbon += weight * data.biomass_emission_factor_t_per_mwh_fuel * fuel_use * data.carbon_price_cny_per_t
                source_variable_om += weight * source.variable_om_cny_per_mwh_th * q_total
                for l in model.L:
                    exergy += (
                        weight
                        * data.exergy_penalty_cny_per_mwh
                        * source.exergy_loss_coeff_by_grade[l]
                        * model.q[s, l, t]
                    )

        comfort_penalty = sum(
            data.comfort_slack_penalty_cny_per_c_h
            * (model.comfort_low_slack[b, t] + model.comfort_high_slack[b, t])
            * time_weight[min(int(t), len(time_weight) - 1)]
            for b in model.B
            for t in model.TSOC
        )
        unmet_penalty = sum(
            data.unmet_heat_penalty_cny_per_mwh * model.unmet_mid[b, t]
            * time_weight[int(t)]
            for b in model.B
            for t in model.T
        )
        return (
            source_capex
            + source_fixed_om
            + storage_capex
            + storage_om
            + site_cost
            + pipe_capex
            + pipe_fixed_om
            + data.operation_weight * (operating + carbon + exergy + source_variable_om + unmet_penalty + comfort_penalty)
        )

    m.objective = pyo.Objective(rule=objective_rule, sense=pyo.minimize)
    m._cp202611_data = data
    m._cp202611_distance_allowed = allowed_pairs
    m._cp202611_edge_distance_km = edge_distance_km
    m._cp202611_edge_loss_fraction = edge_loss_fraction
    m._cp202611_edge_capacity_mw = edge_capacity_mw
    m._cp202611_fixed_decision = fixed_decision
    return m


def solve_mvp(data: MVPScenario, tee: bool = False, fixed_decision: FixedPlanningDecision | None = None) -> SolveResult:
    model = build_mvp_model(data, fixed_decision=fixed_decision)
    solver = pyo.SolverFactory("appsi_highs")
    if not solver.available(exception_flag=False):
        solver = pyo.SolverFactory("highs")
    result = solver.solve(model, tee=tee)
    termination = str(result.solver.termination_condition)
    if termination.lower() not in {"optimal", "feasible"}:
        raise RuntimeError(f"Solver did not find a usable solution: {termination}")
    return extract_solution(model, termination)


def extract_fixed_decision(result: SolveResult) -> FixedPlanningDecision:
    capacities = {
        str(row.source_id): float(row.capacity_mw)
        for row in result.source_capacity.itertuples(index=False)
        if str(row.source_id) != "storage"
    }
    storage_capacity = float(
        result.source_capacity[result.source_capacity["source_id"] == "storage"]["capacity_mw"].iloc[0]
    )
    site_open = {
        str(site_id): int(group["site_open"].max())
        for site_id, group in result.spatial_assignment.groupby("site_id")
    }
    assignment = {
        (str(row.building_id), str(row.site_id)): int(row.assigned)
        for row in result.spatial_assignment.itertuples(index=False)
    }
    return FixedPlanningDecision(
        source_capacity_mw=capacities,
        storage_capacity_mwh=storage_capacity,
        site_open=site_open,
        assignment=assignment,
    )


def value(x: Any) -> float:
    return float(pyo.value(x))


def extract_solution(model: pyo.ConcreteModel, termination: str = "unknown") -> SolveResult:
    data: MVPScenario = model._cp202611_data
    sources = {s.source_id: s for s in data.sources}
    buildings = {b.building_id: b for b in data.buildings}

    dispatch_rows = []
    for t in model.T:
        for s in model.S:
            for l in model.L:
                dispatch_rows.append(
                    {
                        "hour": int(t),
                        "source_id": str(s),
                        "grade": str(l),
                        "heat_mw": value(model.q[s, l, t]),
                    }
                )
        dispatch_rows.append({"hour": int(t), "source_id": "storage", "grade": "low", "heat_mw": value(model.discharge[t])})
        dispatch_rows.append({"hour": int(t), "source_id": "storage_charge", "grade": "low", "heat_mw": -value(model.charge[t])})
    dispatch = pd.DataFrame(dispatch_rows)

    temp_rows = []
    for b in model.B:
        for t in model.TSOC:
            temp_rows.append(
                {
                    "building_id": str(b),
                    "hour": int(t),
                    "indoor_temp_c": value(model.tin[b, t]),
                    "low_slack_c": value(model.comfort_low_slack[b, t]),
                    "high_slack_c": value(model.comfort_high_slack[b, t]),
                }
            )
    indoor = pd.DataFrame(temp_rows)

    storage = pd.DataFrame(
        {
            "hour": [int(t) for t in model.TSOC],
            "soc_mwh": [value(model.soc[t]) for t in model.TSOC],
        }
    )
    hourly_storage = pd.DataFrame(
        {
            "hour": [int(t) for t in model.T],
            "charge_mw": [value(model.charge[t]) for t in model.T],
            "discharge_mw": [value(model.discharge[t]) for t in model.T],
        }
    )
    storage = storage.merge(hourly_storage, on="hour", how="left")

    assignment_rows = []
    for b in model.B:
        for j in model.J:
            building = buildings[str(b)]
            site = next(s for s in data.candidate_sites if s.site_id == str(j))
            assignment_rows.append(
                {
                    "building_id": str(b),
                    "site_id": str(j),
                    "assigned": round(value(model.assign[b, j])),
                    "site_open": round(value(model.site_open[j])),
                    "distance_km": distance_km(building.lon, building.lat, site.lon, site.lat),
                    "max_radius_km": site.max_radius_km,
                }
            )
    spatial = pd.DataFrame(assignment_rows)

    edge_rows = []
    edge_distance_km: dict[tuple[str, str], float] = model._cp202611_edge_distance_km
    edge_loss_fraction: dict[tuple[str, str], float] = model._cp202611_edge_loss_fraction
    edge_capacity_mw: dict[tuple[str, str], float] = model._cp202611_edge_capacity_mw
    for b in model.B:
        for j in model.J:
            assigned = round(value(model.assign[b, j]))
            for t in model.T:
                low_delivered = value(model.edge_low[b, j, t])
                mid_delivered = value(model.edge_mid[b, j, t])
                loss_fraction = edge_loss_fraction[(str(b), str(j))]
                edge_rows.append(
                    {
                        "building_id": str(b),
                        "site_id": str(j),
                        "hour": int(t),
                        "assigned": assigned,
                        "distance_km": edge_distance_km[(str(b), str(j))],
                        "capacity_mw": edge_capacity_mw[(str(b), str(j))],
                        "loss_fraction": loss_fraction,
                        "low_delivered_mw": low_delivered,
                        "mid_delivered_mw": mid_delivered,
                        "source_side_mw": (low_delivered + mid_delivered) * (1.0 + loss_fraction),
                        "loss_mw": (low_delivered + mid_delivered) * loss_fraction,
                    }
                )
    network_edges = pd.DataFrame(edge_rows)

    caps = pd.DataFrame(
        [
            {
                "source_id": str(s),
                "capacity_mw": value(model.cap_source[s]),
                "fuel": sources[str(s)].fuel,
            }
            for s in model.S
        ]
        + [{"source_id": "storage", "capacity_mw": value(model.cap_storage), "fuel": "thermal"}]
    )

    return SolveResult(
        model=model,
        termination_condition=termination,
        objective_value=value(model.objective),
        dispatch=dispatch,
        indoor_temperature=indoor,
        storage=storage,
        spatial_assignment=spatial,
        network_edges=network_edges,
        source_capacity=caps,
    )
