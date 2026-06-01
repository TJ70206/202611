from __future__ import annotations

from dataclasses import dataclass

import pyomo.environ as pyo

from cp202611.optimization.mvp_model import SolveResult


@dataclass(frozen=True)
class MVPDiagnostics:
    max_low_balance_abs_mw: float
    max_mid_balance_abs_mw: float
    soc_closure_abs_mwh: float
    max_comfort_slack_c: float
    storage_charged_in_offpeak_mwh: float
    storage_discharged_in_peak_mwh: float
    network_loss_mwh: float
    total_unmet_mid_mwh: float


def compute_mvp_diagnostics(result: SolveResult) -> MVPDiagnostics:
    model = result.model
    data = model._cp202611_data
    buildings = {b.building_id: b for b in data.buildings}

    max_low = 0.0
    max_mid = 0.0
    network_loss = 0.0
    unmet_mid = 0.0
    edge_loss_fraction = model._cp202611_edge_loss_fraction
    for t in model.T:
        low_supply = sum(pyo.value(model.q[s, "low", t]) for s in model.S) + pyo.value(model.discharge[t])
        low_use = (
            sum(
                pyo.value(model.edge_low[b, j, t]) * (1.0 + edge_loss_fraction[(str(b), str(j))])
                for b in model.B
                for j in model.J
            )
            + pyo.value(model.charge[t])
        )
        max_low = max(max_low, abs(low_supply - low_use))

        mid_supply = sum(pyo.value(model.q[s, "mid", t]) for s in model.S)
        mid_use = sum(
            pyo.value(model.edge_mid[b, j, t]) * (1.0 + edge_loss_fraction[(str(b), str(j))])
            for b in model.B
            for j in model.J
        )
        max_mid = max(max_mid, abs(mid_supply - mid_use))
        network_loss += sum(
            (pyo.value(model.edge_low[b, j, t]) + pyo.value(model.edge_mid[b, j, t]))
            * edge_loss_fraction[(str(b), str(j))]
            for b in model.B
            for j in model.J
        )
        unmet_mid += sum(pyo.value(model.unmet_mid[b, t]) for b in model.B)

    if data.storage_cycle_block_size_h is None:
        soc_closure = abs(pyo.value(model.soc[len(data.hours)]) - pyo.value(model.soc[0]))
    else:
        soc_closure = max(
            abs(pyo.value(model.soc[start + data.storage_cycle_block_size_h]) - pyo.value(model.soc[start]))
            for start in range(0, len(data.hours), data.storage_cycle_block_size_h)
        )
    max_slack = max(
        max(result.indoor_temperature["low_slack_c"].max(), result.indoor_temperature["high_slack_c"].max()),
        0.0,
    )
    offpeak_hours = [h for h, mult in enumerate(data.electricity_price_multiplier) if mult <= 0.31]
    peak_hours = [h for h, mult in enumerate(data.electricity_price_multiplier) if mult >= 1.50]
    storage = result.storage.set_index("hour")
    charged = float(storage.loc[offpeak_hours, "charge_mw"].fillna(0).sum())
    discharged = float(storage.loc[peak_hours, "discharge_mw"].fillna(0).sum())

    return MVPDiagnostics(
        max_low_balance_abs_mw=max_low,
        max_mid_balance_abs_mw=max_mid,
        soc_closure_abs_mwh=soc_closure,
        max_comfort_slack_c=max_slack,
        storage_charged_in_offpeak_mwh=charged,
        storage_discharged_in_peak_mwh=discharged,
        network_loss_mwh=network_loss,
        total_unmet_mid_mwh=unmet_mid,
    )
