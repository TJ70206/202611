from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from cp202611.schema import Building, CandidateSite, HeatSource, MVPScenario, StorageSpec


def write_processed_dataset(data: MVPScenario, root: Path) -> None:
    """Write a scenario to the canonical processed-data layout."""
    root.mkdir(parents=True, exist_ok=True)

    (root / "scenario.yaml").write_text(
        yaml.safe_dump(
            {
                "scenario_id": data.scenario_id,
                "electricity_base_price_cny_per_mwh": data.electricity_base_price_cny_per_mwh,
                "gas_price_cny_per_mwh_fuel": data.gas_price_cny_per_mwh_fuel,
                "biomass_price_cny_per_mwh_fuel": data.biomass_price_cny_per_mwh_fuel,
                "gas_emission_factor_t_per_mwh_fuel": data.gas_emission_factor_t_per_mwh_fuel,
                "biomass_emission_factor_t_per_mwh_fuel": data.biomass_emission_factor_t_per_mwh_fuel,
                "carbon_price_cny_per_t": data.carbon_price_cny_per_t,
                "capex_recovery_factor": data.capex_recovery_factor,
                "operation_weight": data.operation_weight,
                "exergy_penalty_cny_per_mwh": data.exergy_penalty_cny_per_mwh,
                "comfort_slack_penalty_cny_per_c_h": data.comfort_slack_penalty_cny_per_c_h,
                "unmet_heat_penalty_cny_per_mwh": data.unmet_heat_penalty_cny_per_mwh,
                "max_open_sites": data.max_open_sites,
                "pipe_loss_fraction_per_km": data.pipe_loss_fraction_per_km,
                "pipe_capex_cny_per_mw_km": data.pipe_capex_cny_per_mw_km,
                "pipe_fixed_om_fraction": data.pipe_fixed_om_fraction,
                "pipe_capacity_margin": data.pipe_capacity_margin,
                "storage_cycle_block_size_h": data.storage_cycle_block_size_h,
            },
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )

    pd.DataFrame(
        {
            "hour": data.hours,
            "outdoor_temperature_c": data.outdoor_temperature_c,
            "electricity_price_multiplier": data.electricity_price_multiplier,
            "grid_carbon_factor_t_per_mwh": data.grid_carbon_factor_t_per_mwh,
            "grid_import_limit_mw": data.grid_import_limit_mw,
            "time_weight": data.time_weight or [1.0 for _ in data.hours],
        }
    ).to_csv(root / "time_series.csv", index=False, encoding="utf-8-sig")

    building_rows: list[dict[str, Any]] = []
    mid_rows: list[dict[str, Any]] = []
    for building in data.buildings:
        row = building.model_dump(exclude={"mid_temp_demand_mw"})
        building_rows.append(row)
        for hour, demand in zip(data.hours, building.mid_temp_demand_mw):
            mid_rows.append({"building_id": building.building_id, "hour": hour, "mid_temp_demand_mw": demand})
    pd.DataFrame(building_rows).to_csv(root / "buildings.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(mid_rows).to_csv(root / "building_mid_temp_demand.csv", index=False, encoding="utf-8-sig")

    source_rows = []
    for source in data.sources:
        row = source.model_dump()
        row["allowed_grades"] = "|".join(source.allowed_grades)
        row["exergy_loss_coeff_by_grade"] = json.dumps(source.exergy_loss_coeff_by_grade, ensure_ascii=False)
        source_rows.append(row)
    pd.DataFrame(source_rows).to_csv(root / "heat_sources.csv", index=False, encoding="utf-8-sig")

    pd.DataFrame([site.model_dump() for site in data.candidate_sites]).to_csv(
        root / "candidate_sites.csv", index=False, encoding="utf-8-sig"
    )
    (root / "storage.yaml").write_text(
        yaml.safe_dump(data.storage.model_dump(), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def load_processed_dataset(root: Path) -> MVPScenario:
    """Load a scenario from canonical processed CSV/YAML files."""
    scenario_cfg = yaml.safe_load((root / "scenario.yaml").read_text(encoding="utf-8"))
    time_series = pd.read_csv(root / "time_series.csv")
    buildings_df = pd.read_csv(root / "buildings.csv")
    mid_df = pd.read_csv(root / "building_mid_temp_demand.csv")
    sources_df = pd.read_csv(root / "heat_sources.csv")
    sites_df = pd.read_csv(root / "candidate_sites.csv")
    storage_cfg = yaml.safe_load((root / "storage.yaml").read_text(encoding="utf-8"))

    hours = [int(h) for h in time_series["hour"].tolist()]
    mid_lookup = {
        str(building_id): group.sort_values("hour")["mid_temp_demand_mw"].astype(float).tolist()
        for building_id, group in mid_df.groupby("building_id")
    }

    buildings = []
    for row in buildings_df.to_dict(orient="records"):
        row["building_id"] = str(row["building_id"])
        row["mid_temp_demand_mw"] = mid_lookup[row["building_id"]]
        buildings.append(Building(**row))

    sources = []
    for row in sources_df.to_dict(orient="records"):
        row["source_id"] = str(row["source_id"])
        row["allowed_grades"] = str(row["allowed_grades"]).split("|")
        row["exergy_loss_coeff_by_grade"] = json.loads(row["exergy_loss_coeff_by_grade"])
        if pd.isna(row.get("base_cop")):
            row["base_cop"] = None
        sources.append(HeatSource(**row))

    sites = [CandidateSite(**row) for row in sites_df.to_dict(orient="records")]

    return MVPScenario(
        **scenario_cfg,
        hours=hours,
        outdoor_temperature_c=time_series["outdoor_temperature_c"].astype(float).tolist(),
        electricity_price_multiplier=time_series["electricity_price_multiplier"].astype(float).tolist(),
        grid_carbon_factor_t_per_mwh=time_series["grid_carbon_factor_t_per_mwh"].astype(float).tolist(),
        grid_import_limit_mw=time_series["grid_import_limit_mw"].astype(float).tolist(),
        time_weight=time_series["time_weight"].astype(float).tolist() if "time_weight" in time_series.columns else None,
        sources=sources,
        buildings=buildings,
        candidate_sites=sites,
        storage=StorageSpec(**storage_cfg),
    )
