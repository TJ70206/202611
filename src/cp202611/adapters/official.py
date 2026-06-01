from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from cp202611.dataio import write_processed_dataset
from cp202611.schema import Building, CandidateSite, HeatSource, MVPScenario, StorageSpec
from cp202611.validation import ValidationReport, validate_scenario


def preprocess_official_data(raw_dir: Path, mapping_path: Path, output_dir: Path) -> ValidationReport:
    """Convert organizer/public raw tables into the canonical processed layout."""
    scenario = load_official_mapped_dataset(raw_dir, mapping_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_processed_dataset(scenario, output_dir)
    report = validate_scenario(scenario)
    report.to_dataframe().to_csv(output_dir / "validation_issues.csv", index=False, encoding="utf-8-sig")
    (output_dir / "validation_report.md").write_text(report.to_markdown(), encoding="utf-8")
    return report


def load_official_mapped_dataset(raw_dir: Path, mapping_path: Path) -> MVPScenario:
    """Load a scenario from arbitrary raw tables using a YAML field mapping."""
    config = yaml.safe_load(mapping_path.read_text(encoding="utf-8"))
    scenario_cfg = dict(config["scenario"])
    time_series = _mapped_table(raw_dir, config["time_series"])
    time_series = _sort_unique_hours(time_series, "time_series")
    hours = time_series["hour"].astype(int).tolist()

    buildings_df = _mapped_table(raw_dir, config["buildings"])
    mid_df = _mapped_table(raw_dir, config["building_mid_temp_demand"])
    mid_df = _sort_unique_hours_by_group(mid_df, "building_id", "building_mid_temp_demand")
    sources_df = _mapped_table(raw_dir, config["heat_sources"])
    sites_df = _mapped_table(raw_dir, config["candidate_sites"])
    storage_cfg = dict(config["storage"])

    mid_lookup = _build_hour_aligned_mid_lookup(mid_df, hours)

    buildings: list[Building] = []
    for row in buildings_df.to_dict(orient="records"):
        payload = {key: _clean_scalar(value) for key, value in row.items()}
        building_id = str(payload["building_id"])
        if building_id not in mid_lookup:
            raise ValueError(f"building {building_id!r} is missing mid-temperature demand rows")
        payload["building_id"] = building_id
        payload["mid_temp_demand_mw"] = mid_lookup[building_id]
        buildings.append(Building(**payload))

    sources: list[HeatSource] = []
    for row in sources_df.to_dict(orient="records"):
        payload = {key: _clean_scalar(value) for key, value in row.items()}
        payload["source_id"] = str(payload["source_id"])
        payload["fuel"] = _normalize_fuel(payload["fuel"])
        payload["allowed_grades"] = _normalize_grades(payload["allowed_grades"])
        if "exergy_loss_coeff_by_grade" in payload and payload["exergy_loss_coeff_by_grade"] not in ("", None):
            payload["exergy_loss_coeff_by_grade"] = _parse_exergy_map(payload["exergy_loss_coeff_by_grade"])
        else:
            payload["exergy_loss_coeff_by_grade"] = {
                "low": float(payload.pop("exergy_loss_low")),
                "mid": float(payload.pop("exergy_loss_mid")),
            }
        if payload.get("base_cop") in ("", None):
            payload["base_cop"] = None
        sources.append(HeatSource(**payload))

    sites = [
        CandidateSite(**{key: _clean_scalar(value) for key, value in row.items()})
        for row in sites_df.to_dict(orient="records")
    ]

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


def _mapped_table(raw_dir: Path, spec: dict[str, Any]) -> pd.DataFrame:
    frame = _read_table(raw_dir / spec["file"], sheet_name=spec.get("sheet_name"))
    frame = frame.copy()
    frame.columns = [str(column).strip() for column in frame.columns]
    columns: dict[str, str | list[str]] = spec.get("columns", {})
    defaults: dict[str, Any] = spec.get("defaults", {})

    mapped: dict[str, pd.Series] = {}
    for canonical, raw_name in columns.items():
        resolved = _resolve_raw_column(frame.columns, raw_name, spec["file"], canonical)
        mapped[canonical] = frame[resolved]
    output = pd.DataFrame(mapped)
    for canonical, value in defaults.items():
        if canonical not in output.columns:
            output[canonical] = value
    return output


def _resolve_raw_column(
    columns: pd.Index,
    raw_name: str | list[str],
    file_name: str,
    canonical: str,
) -> str:
    candidates = raw_name if isinstance(raw_name, list) else [raw_name]
    available = list(columns)
    normalized_lookup = {_normalize_column_name(column): column for column in available}
    for candidate in candidates:
        candidate_text = str(candidate).strip()
        if candidate_text in columns:
            return candidate_text
        normalized = _normalize_column_name(candidate_text)
        if normalized in normalized_lookup:
            return normalized_lookup[normalized]
    raise ValueError(
        f"raw column {raw_name!r} for {canonical!r} not found in {file_name}; "
        f"available columns: {available}"
    )


def _normalize_column_name(value: Any) -> str:
    return str(value).replace("\ufeff", "").strip().lower()


def _sort_unique_hours(frame: pd.DataFrame, table_name: str) -> pd.DataFrame:
    if "hour" not in frame.columns:
        raise ValueError(f"{table_name} is missing canonical 'hour' column")
    output = frame.copy()
    output["hour"] = output["hour"].astype(int)
    duplicates = output.loc[output["hour"].duplicated(), "hour"].astype(int).tolist()
    if duplicates:
        raise ValueError(f"{table_name} contains duplicate hour rows: {duplicates[:5]}")
    return output.sort_values("hour").reset_index(drop=True)


def _sort_unique_hours_by_group(frame: pd.DataFrame, group_column: str, table_name: str) -> pd.DataFrame:
    if group_column not in frame.columns:
        raise ValueError(f"{table_name} is missing canonical {group_column!r} column")
    output = frame.copy()
    output["hour"] = output["hour"].astype(int)
    duplicate_mask = output.duplicated([group_column, "hour"])
    if duplicate_mask.any():
        sample = output.loc[duplicate_mask, [group_column, "hour"]].head(5).to_dict(orient="records")
        raise ValueError(f"{table_name} contains duplicate group-hour rows: {sample}")
    return output.sort_values([group_column, "hour"]).reset_index(drop=True)


def _build_hour_aligned_mid_lookup(mid_df: pd.DataFrame, hours: list[int]) -> dict[str, list[float]]:
    hour_index = pd.Index(hours, name="hour")
    lookup: dict[str, list[float]] = {}
    for building_id, group in mid_df.groupby("building_id"):
        indexed = group.set_index("hour").reindex(hour_index)
        missing_hours = indexed[indexed["mid_temp_demand_mw"].isna()].index.astype(int).tolist()
        if missing_hours:
            raise ValueError(
                f"building {str(building_id)!r} mid-temperature demand is missing hours: "
                f"{missing_hours[:10]}"
            )
        lookup[str(building_id)] = indexed["mid_temp_demand_mw"].astype(float).tolist()
    return lookup


def _read_table(path: Path, sheet_name: str | None = None) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".xlsx", ".xls"}:
        try:
            return pd.read_excel(path, sheet_name=sheet_name or 0)
        except ImportError as exc:
            raise ImportError("Reading Excel raw data requires openpyxl/xlrd in the CP202611 environment.") from exc
    raise ValueError(f"unsupported raw table format: {path}")


def _clean_scalar(value: Any) -> Any:
    if pd.isna(value):
        return None
    return value


def _normalize_grades(value: Any) -> list[str]:
    if isinstance(value, list):
        parts = value
    else:
        text = str(value).replace("，", "|").replace(",", "|").replace("、", "|").replace("/", "|")
        parts = [part.strip() for part in text.split("|") if part.strip()]
    mapped = []
    for part in parts:
        token = str(part).strip().lower()
        if token in {"low", "低温", "低品位", "space", "space_heating"}:
            mapped.append("low")
        elif token in {"mid", "中温", "中品位", "dhw", "commercial"}:
            mapped.append("mid")
        else:
            raise ValueError(f"unknown temperature grade: {part!r}")
    return sorted(set(mapped))


def _normalize_fuel(value: Any) -> str:
    token = str(value).strip().lower()
    if token in {"electricity", "electric", "power", "电", "电力"}:
        return "electricity"
    if token in {"gas", "natural_gas", "天然气", "燃气"}:
        return "gas"
    if token in {"biomass", "bio", "生物质"}:
        return "biomass"
    raise ValueError(f"unknown fuel type: {value!r}")


def _parse_exergy_map(value: Any) -> dict[str, float]:
    if isinstance(value, dict):
        return {str(k): float(v) for k, v in value.items()}
    parsed = json.loads(str(value))
    return {str(k): float(v) for k, v in parsed.items()}
