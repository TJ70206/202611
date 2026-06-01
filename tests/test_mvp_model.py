from __future__ import annotations

import pytest
import pyomo.environ as pyo

from cp202611.analysis.diagnostics import compute_mvp_diagnostics
from cp202611.optimization.mvp_model import distance_km, route_distance_km


def test_mvp_solves_to_optimality(solved_mvp):
    assert solved_mvp.termination_condition.lower() == "optimal"


def test_heat_balance_and_soc_closure(solved_mvp):
    diagnostics = compute_mvp_diagnostics(solved_mvp)
    assert diagnostics.max_low_balance_abs_mw < 1e-6
    assert diagnostics.max_mid_balance_abs_mw < 1e-6
    assert diagnostics.soc_closure_abs_mwh < 1e-6


def test_temperature_grade_hard_constraints(solved_mvp):
    model = solved_mvp.model
    for hour in model.T:
        assert abs(pyo.value(model.q["ashp_low", "mid", hour])) < 1e-8


def test_spatial_assignments_respect_radius(solved_mvp):
    assigned = solved_mvp.spatial_assignment[solved_mvp.spatial_assignment["assigned"] == 1]
    assert not assigned.empty
    assert (assigned["distance_km"] <= assigned["max_radius_km"] + 1e-9).all()
    assert assigned.groupby("building_id")["assigned"].sum().eq(1).all()


def test_network_edge_losses_are_explicit_and_positive(solved_mvp):
    diagnostics = compute_mvp_diagnostics(solved_mvp)
    assert diagnostics.network_loss_mwh > 0.01
    assert "loss_mw" in solved_mvp.network_edges.columns
    assert solved_mvp.network_edges["loss_mw"].sum() > 0.01


def test_unassigned_network_edges_carry_no_flow(solved_mvp):
    unassigned = solved_mvp.network_edges[solved_mvp.network_edges["assigned"] == 0]
    assert not unassigned.empty
    assert unassigned["low_delivered_mw"].abs().max() < 1e-8
    assert unassigned["mid_delivered_mw"].abs().max() < 1e-8


def test_storage_responds_to_time_of_use_price(solved_mvp):
    diagnostics = compute_mvp_diagnostics(solved_mvp)
    assert diagnostics.storage_charged_in_offpeak_mwh > 0.05
    assert diagnostics.storage_discharged_in_peak_mwh > 0.05


def test_gas_boiler_participates_as_firm_peaking_source(solved_mvp):
    capacity = solved_mvp.source_capacity.set_index("source_id").loc["gas_boiler", "capacity_mw"]
    gas_dispatch = solved_mvp.dispatch[solved_mvp.dispatch["source_id"] == "gas_boiler"]["heat_mw"].sum()
    assert capacity > 0.05
    assert gas_dispatch > 0.05


def test_comfort_slack_is_not_used_in_base_case(solved_mvp):
    diagnostics = compute_mvp_diagnostics(solved_mvp)
    assert diagnostics.max_comfort_slack_c < 1e-5


def test_distance_km_is_reasonable():
    d = distance_km(117.02, 36.66, 117.04, 36.66)
    assert 1.6 < d < 1.9


def test_route_distance_applies_planning_detour_factor():
    straight = distance_km(117.02, 36.66, 117.04, 36.66)
    routed = route_distance_km(117.02, 36.66, 117.04, 36.66, route_factor=1.3)

    assert routed == pytest.approx(straight * 1.3)
    assert routed > straight


def test_indoor_temperature_has_soft_target_in_base_case(solved_mvp):
    indoor = solved_mvp.indoor_temperature
    active_hours = indoor[indoor["hour"] < indoor["hour"].max()]

    assert active_hours["indoor_temp_c"].mean() > 19.2
    assert active_hours["indoor_temp_c"].min() >= 18.0 - 1e-6
