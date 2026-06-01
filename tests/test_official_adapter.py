from __future__ import annotations

import pandas as pd
import pytest
import yaml

from cp202611.adapters.official import load_official_mapped_dataset, preprocess_official_data
from cp202611.dataio import load_processed_dataset
from cp202611.synthetic import create_synthetic_mvp


def _write_fake_official_raw(raw_dir, scenario):
    raw_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "时段": scenario.hours,
            "室外温度": scenario.outdoor_temperature_c,
            "电价倍率": scenario.electricity_price_multiplier,
            "电网碳因子": scenario.grid_carbon_factor_t_per_mwh,
            "电网接入上限": scenario.grid_import_limit_mw,
        }
    ).to_csv(raw_dir / "official_time.csv", index=False, encoding="utf-8-sig")

    pd.DataFrame(
        [
            {
                "建筑编号": building.building_id,
                "经度": building.lon,
                "纬度": building.lat,
                "面积": building.floor_area_m2,
                "热损系数": building.heat_loss_mw_per_c,
                "热容": building.thermal_capacity_mwh_per_c,
                "初始室温": building.initial_indoor_temp_c,
                "舒适下限": building.comfort_min_c,
                "舒适上限": building.comfort_max_c,
                "内扰": building.internal_gain_mw,
                "峰值低温负荷": building.peak_heat_mw,
            }
            for building in scenario.buildings
        ]
    ).to_csv(raw_dir / "official_buildings.csv", index=False, encoding="utf-8-sig")

    rows = []
    for building in scenario.buildings:
        for hour, demand in zip(scenario.hours, building.mid_temp_demand_mw):
            rows.append({"建筑编号": building.building_id, "时段": hour, "中温需求": demand})
    pd.DataFrame(rows).to_csv(raw_dir / "official_mid.csv", index=False, encoding="utf-8-sig")

    pd.DataFrame(
        [
            {
                "热源编号": source.source_id,
                "燃料": source.fuel,
                "可供温度": "|".join(source.allowed_grades),
                "最大容量": source.max_capacity_mw,
                "投资": source.capex_cny_per_mw,
                "固定运维率": source.fixed_om_fraction,
                "效率": source.efficiency,
                "基础COP": source.base_cop,
                "变动运维": source.variable_om_cny_per_mwh_th,
                "低温㶲损": source.exergy_loss_coeff_by_grade["low"],
                "中温㶲损": source.exergy_loss_coeff_by_grade["mid"],
            }
            for source in scenario.sources
        ]
    ).to_csv(raw_dir / "official_sources.csv", index=False, encoding="utf-8-sig")

    pd.DataFrame(
        [
            {
                "站点编号": site.site_id,
                "经度": site.lon,
                "纬度": site.lat,
                "半径": site.max_radius_km,
                "服务能力": site.service_capacity_mw,
                "固定成本": site.fixed_cost_cny,
            }
            for site in scenario.candidate_sites
        ]
    ).to_csv(raw_dir / "official_sites.csv", index=False, encoding="utf-8-sig")


def _write_mapping(mapping_path, scenario):
    mapping = {
        "scenario": {
            "scenario_id": "fake_official_mapping_case",
            "electricity_base_price_cny_per_mwh": scenario.electricity_base_price_cny_per_mwh,
            "gas_price_cny_per_mwh_fuel": scenario.gas_price_cny_per_mwh_fuel,
            "biomass_price_cny_per_mwh_fuel": scenario.biomass_price_cny_per_mwh_fuel,
            "gas_emission_factor_t_per_mwh_fuel": scenario.gas_emission_factor_t_per_mwh_fuel,
            "biomass_emission_factor_t_per_mwh_fuel": scenario.biomass_emission_factor_t_per_mwh_fuel,
            "carbon_price_cny_per_t": scenario.carbon_price_cny_per_t,
            "capex_recovery_factor": scenario.capex_recovery_factor,
            "operation_weight": scenario.operation_weight,
            "exergy_penalty_cny_per_mwh": scenario.exergy_penalty_cny_per_mwh,
            "comfort_slack_penalty_cny_per_c_h": scenario.comfort_slack_penalty_cny_per_c_h,
            "unmet_heat_penalty_cny_per_mwh": scenario.unmet_heat_penalty_cny_per_mwh,
            "max_open_sites": scenario.max_open_sites,
            "pipe_loss_fraction_per_km": scenario.pipe_loss_fraction_per_km,
            "pipe_capex_cny_per_mw_km": scenario.pipe_capex_cny_per_mw_km,
            "pipe_fixed_om_fraction": scenario.pipe_fixed_om_fraction,
            "pipe_capacity_margin": scenario.pipe_capacity_margin,
            "storage_cycle_block_size_h": None,
        },
        "time_series": {
            "file": "official_time.csv",
            "columns": {
                "hour": "时段",
                "outdoor_temperature_c": "室外温度",
                "electricity_price_multiplier": "电价倍率",
                "grid_carbon_factor_t_per_mwh": "电网碳因子",
                "grid_import_limit_mw": "电网接入上限",
            },
        },
        "buildings": {
            "file": "official_buildings.csv",
            "columns": {
                "building_id": "建筑编号",
                "lon": "经度",
                "lat": "纬度",
                "floor_area_m2": "面积",
                "heat_loss_mw_per_c": "热损系数",
                "thermal_capacity_mwh_per_c": "热容",
                "initial_indoor_temp_c": "初始室温",
                "comfort_min_c": "舒适下限",
                "comfort_max_c": "舒适上限",
                "internal_gain_mw": "内扰",
                "peak_heat_mw": "峰值低温负荷",
            },
        },
        "building_mid_temp_demand": {
            "file": "official_mid.csv",
            "columns": {"building_id": "建筑编号", "hour": "时段", "mid_temp_demand_mw": "中温需求"},
        },
        "heat_sources": {
            "file": "official_sources.csv",
            "columns": {
                "source_id": "热源编号",
                "fuel": "燃料",
                "allowed_grades": "可供温度",
                "max_capacity_mw": "最大容量",
                "capex_cny_per_mw": "投资",
                "fixed_om_fraction": "固定运维率",
                "efficiency": "效率",
                "base_cop": "基础COP",
                "variable_om_cny_per_mwh_th": "变动运维",
                "exergy_loss_low": "低温㶲损",
                "exergy_loss_mid": "中温㶲损",
            },
        },
        "candidate_sites": {
            "file": "official_sites.csv",
            "columns": {
                "site_id": "站点编号",
                "lon": "经度",
                "lat": "纬度",
                "max_radius_km": "半径",
                "service_capacity_mw": "服务能力",
                "fixed_cost_cny": "固定成本",
            },
        },
        "storage": scenario.storage.model_dump(),
    }
    mapping_path.write_text(yaml.safe_dump(mapping, allow_unicode=True, sort_keys=False), encoding="utf-8")


def test_official_mapping_loads_arbitrary_raw_columns(tmp_path):
    scenario = create_synthetic_mvp()
    raw_dir = tmp_path / "raw"
    mapping_path = tmp_path / "field_mapping.yaml"
    _write_fake_official_raw(raw_dir, scenario)
    _write_mapping(mapping_path, scenario)

    mapped = load_official_mapped_dataset(raw_dir, mapping_path)

    assert mapped.scenario_id == "fake_official_mapping_case"
    assert len(mapped.hours) == len(scenario.hours)
    assert mapped.buildings[0].building_id == scenario.buildings[0].building_id
    assert mapped.sources[0].allowed_grades == scenario.sources[0].allowed_grades


def test_preprocess_official_writes_canonical_dataset_and_validation_report(tmp_path):
    scenario = create_synthetic_mvp()
    raw_dir = tmp_path / "raw"
    mapping_path = tmp_path / "field_mapping.yaml"
    output_dir = tmp_path / "processed"
    _write_fake_official_raw(raw_dir, scenario)
    _write_mapping(mapping_path, scenario)

    report = preprocess_official_data(raw_dir, mapping_path, output_dir)
    loaded = load_processed_dataset(output_dir)

    assert report.is_usable
    assert report.error_count == 0
    assert (output_dir / "validation_report.md").exists()
    assert loaded.scenario_id == "fake_official_mapping_case"
    assert loaded.buildings[1].mid_temp_demand_mw == scenario.buildings[1].mid_temp_demand_mw


def test_official_adapter_reads_excel_raw_table(tmp_path):
    scenario = create_synthetic_mvp()
    raw_dir = tmp_path / "raw"
    mapping_path = tmp_path / "field_mapping.yaml"
    _write_fake_official_raw(raw_dir, scenario)
    _write_mapping(mapping_path, scenario)

    time_frame = pd.read_csv(raw_dir / "official_time.csv")
    time_frame.to_excel(raw_dir / "official_time.xlsx", index=False)
    mapping = yaml.safe_load(mapping_path.read_text(encoding="utf-8"))
    mapping["time_series"]["file"] = "official_time.xlsx"
    mapping_path.write_text(yaml.safe_dump(mapping, allow_unicode=True, sort_keys=False), encoding="utf-8")

    mapped = load_official_mapped_dataset(raw_dir, mapping_path)

    assert mapped.outdoor_temperature_c == pytest.approx(scenario.outdoor_temperature_c)


def test_official_mapping_sorts_time_series_and_accepts_alias_columns(tmp_path):
    scenario = create_synthetic_mvp()
    raw_dir = tmp_path / "raw"
    mapping_path = tmp_path / "field_mapping.yaml"
    _write_fake_official_raw(raw_dir, scenario)
    _write_mapping(mapping_path, scenario)

    time_frame = pd.read_csv(raw_dir / "official_time.csv")
    time_frame = time_frame.iloc[::-1].reset_index(drop=True)
    time_frame.columns = [f"  {column}  " for column in time_frame.columns]
    time_frame.to_csv(raw_dir / "official_time.csv", index=False, encoding="utf-8-sig")

    mapping = yaml.safe_load(mapping_path.read_text(encoding="utf-8"))
    mapping["time_series"]["columns"] = {
        canonical: [f"not_{raw_name}", f"  {raw_name}  "]
        for canonical, raw_name in mapping["time_series"]["columns"].items()
    }
    mapping_path.write_text(yaml.safe_dump(mapping, allow_unicode=True, sort_keys=False), encoding="utf-8")

    mapped = load_official_mapped_dataset(raw_dir, mapping_path)

    assert mapped.hours == sorted(scenario.hours)
    assert mapped.outdoor_temperature_c == pytest.approx(scenario.outdoor_temperature_c)


def test_official_mapping_rejects_duplicate_time_series_hours(tmp_path):
    scenario = create_synthetic_mvp()
    raw_dir = tmp_path / "raw"
    mapping_path = tmp_path / "field_mapping.yaml"
    _write_fake_official_raw(raw_dir, scenario)
    _write_mapping(mapping_path, scenario)

    time_frame = pd.read_csv(raw_dir / "official_time.csv")
    time_frame = pd.concat([time_frame, time_frame.iloc[[0]]], ignore_index=True)
    time_frame.to_csv(raw_dir / "official_time.csv", index=False, encoding="utf-8-sig")

    with pytest.raises(ValueError, match="duplicate hour"):
        load_official_mapped_dataset(raw_dir, mapping_path)


def test_official_mapping_rejects_missing_building_hourly_demand(tmp_path):
    scenario = create_synthetic_mvp()
    raw_dir = tmp_path / "raw"
    mapping_path = tmp_path / "field_mapping.yaml"
    _write_fake_official_raw(raw_dir, scenario)
    _write_mapping(mapping_path, scenario)

    mid_frame = pd.read_csv(raw_dir / "official_mid.csv")
    first_building = scenario.buildings[0].building_id
    building_column = "建筑编号"
    hour_column = "时段"
    missing_mask = (mid_frame[building_column] == first_building) & (mid_frame[hour_column] == scenario.hours[0])
    mid_frame = mid_frame.loc[~missing_mask]
    mid_frame.to_csv(raw_dir / "official_mid.csv", index=False, encoding="utf-8-sig")

    with pytest.raises(ValueError, match="missing hours"):
        load_official_mapped_dataset(raw_dir, mapping_path)
