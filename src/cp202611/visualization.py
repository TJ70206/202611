from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import pandas as pd
import plotly.graph_objects as go

from cp202611.dataio import load_processed_dataset
from cp202611.schema import MVPScenario


@dataclass(frozen=True)
class FigureRecord:
    file: str
    description: str


@dataclass(frozen=True)
class VisualizationResult:
    output_dir: Path
    figures: list[FigureRecord]

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([asdict(record) for record in self.figures])


BRAND = {
    "ink": "#1F2933",
    "muted": "#6B7280",
    "hairline": "#D7DEE8",
    "grid": "#E8EDF3",
    "blue": "#0F4D92",
    "blue_soft": "#84A9D8",
    "teal": "#2F9E9E",
    "green": "#5FA777",
    "amber": "#E3A23A",
    "red": "#C94C4C",
    "violet": "#7A5DA8",
    "surface": "#FFFFFF",
}

SOURCE_COLORS = {
    "ashp_low": BRAND["blue"],
    "biomass_boiler": BRAND["green"],
    "gas_boiler": BRAND["amber"],
    "electric_boiler": BRAND["red"],
    "storage": BRAND["violet"],
}


def apply_report_style() -> None:
    """Apply a restrained publication/report style for all static figures."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 10,
            "axes.titlesize": 14,
            "axes.titleweight": "semibold",
            "axes.labelsize": 10,
            "axes.edgecolor": BRAND["ink"],
            "axes.linewidth": 0.8,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "legend.frameon": False,
            "xtick.color": BRAND["muted"],
            "ytick.color": BRAND["muted"],
            "text.color": BRAND["ink"],
            "figure.facecolor": BRAND["surface"],
            "axes.facecolor": BRAND["surface"],
        }
    )


def create_report_figures(
    result_dir: Path,
    output_dir: Path,
    dataset_dir: Path | None = None,
    pareto_dir: Path | None = None,
) -> VisualizationResult:
    """Generate report-ready figures from an existing optimization output directory."""
    apply_report_style()
    output_dir.mkdir(parents=True, exist_ok=True)
    scenario = load_processed_dataset(dataset_dir) if dataset_dir is not None else None
    figures: list[FigureRecord] = []

    dispatch_path = _first_existing(
        result_dir,
        ["verification_dispatch.csv", "mvp_dispatch.csv", "solve_dispatch.csv", "week_dispatch.csv"],
    )
    storage_path = _first_existing(result_dir, ["verification_storage.csv", "mvp_storage.csv", "solve_storage.csv"])
    indoor_path = _first_existing(
        result_dir,
        ["verification_indoor_temperature.csv", "mvp_indoor_temperature.csv", "solve_indoor_temperature.csv"],
    )
    capacity_path = _first_existing(result_dir, ["verification_capacity.csv", "mvp_capacity.csv", "solve_capacity.csv"])
    assignment_path = _first_existing(
        result_dir,
        ["planning_spatial_assignment.csv", "mvp_spatial_assignment.csv", "solve_spatial_assignment.csv"],
    )
    network_path = _first_existing(
        result_dir,
        ["verification_network_edges.csv", "mvp_network_edges.csv", "solve_network_edges.csv"],
    )

    if dispatch_path is not None:
        figures.append(_plot_source_stack(pd.read_csv(dispatch_path), output_dir / "source_dispatch_stack.png"))
    if storage_path is not None:
        figures.append(_plot_storage(pd.read_csv(storage_path), output_dir / "storage_soc_dispatch.png"))
    if indoor_path is not None:
        figures.append(_plot_indoor_temperature(pd.read_csv(indoor_path), output_dir / "indoor_temperature_constraints.png", scenario))
    if capacity_path is not None:
        figures.append(_plot_capacity(pd.read_csv(capacity_path), output_dir / "capacity_mix.png"))
    if scenario is not None and assignment_path is not None:
        assignment = pd.read_csv(assignment_path)
        figures.append(_plot_spatial_network_map_html(assignment, scenario, output_dir / "spatial_network_map.html"))
        figures.append(_plot_spatial_network_html(assignment, scenario, output_dir / "spatial_network.html"))
    if dispatch_path is not None and network_path is not None:
        figures.append(
            _plot_energy_sankey_html(
                pd.read_csv(dispatch_path),
                pd.read_csv(network_path),
                output_dir / "energy_sankey.html",
            )
        )
    pareto_path = _first_existing(pareto_dir, ["pareto_front.csv", "pareto_runs.csv"]) if pareto_dir is not None else None
    if pareto_path is not None:
        figures.append(_plot_pareto(pd.read_csv(pareto_path), output_dir / "pareto_cost_carbon_exergy.png"))

    manifest = VisualizationResult(output_dir=output_dir, figures=figures)
    manifest.to_dataframe().to_csv(output_dir / "visualization_manifest.csv", index=False, encoding="utf-8-sig")
    _write_visual_dashboard(output_dir, figures)
    return manifest


def _first_existing(root: Path | None, names: list[str]) -> Path | None:
    if root is None:
        return None
    for name in names:
        path = root / name
        if path.exists():
            return path
    return None


def _plot_source_stack(dispatch: pd.DataFrame, path: Path) -> FigureRecord:
    positive = dispatch[dispatch["heat_mw"] > 1e-9].copy()
    long_horizon = int(positive["hour"].max()) + 1 > 720 if not positive.empty else False
    if long_horizon:
        positive["period"] = positive["hour"] // 24
        period_label = "Day"
        ylabel = "Daily heat / MWh"
        title = "Daily Heat Dispatch by Source"
    else:
        positive["period"] = positive["hour"]
        period_label = "Hour"
        ylabel = "Heat output / MW"
        title = "Hourly Heat Dispatch by Source"
    source_order = [
        "ashp_low",
        "biomass_boiler",
        "gas_boiler",
        "electric_boiler",
        "storage",
    ]
    pivot = (
        positive.groupby(["period", "source_id"], as_index=False)["heat_mw"].sum()
        .pivot(index="period", columns="source_id", values="heat_mw")
        .fillna(0.0)
    )
    columns = [source for source in source_order if source in pivot.columns]
    columns += [source for source in pivot.columns if source not in columns]

    fig, ax = plt.subplots(figsize=(11.2, 5.7))
    colors = [SOURCE_COLORS.get(col, BRAND["blue_soft"]) for col in columns]
    ax.stackplot(pivot.index.to_numpy(), [pivot[col].to_numpy() for col in columns], labels=columns, colors=colors, alpha=0.92)
    charge_frame = dispatch[dispatch["source_id"] == "storage_charge"].copy()
    if long_horizon:
        charge_frame["period"] = charge_frame["hour"] // 24
    else:
        charge_frame["period"] = charge_frame["hour"]
    charge = charge_frame.groupby("period")["heat_mw"].sum().abs()
    if not charge.empty:
        ax.plot(charge.index.to_numpy(), charge.to_numpy(), color=BRAND["ink"], linestyle=(0, (4, 2)), linewidth=1.5, label="storage charge")
    ax.set_title(title, loc="left")
    ax.set_xlabel(period_label)
    ax.set_ylabel(ylabel)
    _clean_axis(ax)
    ax.legend(loc="upper left", ncols=3, fontsize=8, handlelength=1.2, columnspacing=1.2)
    ax.text(1.0, 1.03, "source-network-load balance", transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color=BRAND["muted"])
    fig.tight_layout()
    _save_static(fig, path)
    plt.close(fig)
    return FigureRecord(str(path), "Hourly source dispatch stack with storage charge line.")


def _plot_storage(storage: pd.DataFrame, path: Path) -> FigureRecord:
    fig, ax1 = plt.subplots(figsize=(11.2, 5.2))
    long_horizon = int(storage["hour"].max()) > 720
    if long_horizon:
        frame = storage.copy()
        frame["day"] = frame["hour"] // 24
        soc = frame.groupby("day", as_index=False)["soc_mwh"].agg(["min", "mean", "max"]).reset_index()
        ax1.fill_between(soc["day"].to_numpy(), soc["min"].to_numpy(), soc["max"].to_numpy(), color=BRAND["blue_soft"], alpha=0.22, label="SOC range")
        ax1.plot(soc["day"], soc["mean"], color=BRAND["blue"], linewidth=2.0, label="SOC mean")
        hourly = frame.dropna(subset=["charge_mw", "discharge_mw"]).copy()
        hourly = hourly.groupby("day", as_index=False)[["charge_mw", "discharge_mw"]].sum()
        x_col = "day"
        xlabel = "Day"
        power_label = "Daily charge (+) / discharge (-) MWh"
        title = "Daily Storage SOC Envelope and Charge/Discharge"
    else:
        ax1.plot(storage["hour"], storage["soc_mwh"], color=BRAND["blue"], linewidth=2.0, label="SOC")
        hourly = storage.dropna(subset=["charge_mw", "discharge_mw"])
        x_col = "hour"
        xlabel = "Hour"
        power_label = "Charge (+) / discharge (-) MW"
        title = "Storage SOC and Charge/Discharge"
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel("Storage SOC / MWh", color=BRAND["blue"])
    ax1.tick_params(axis="y", labelcolor="#214E8A")
    _clean_axis(ax1)

    ax2 = ax1.twinx()
    ax2.bar(hourly[x_col], hourly["charge_mw"], color=BRAND["green"], alpha=0.26, label="charge")
    ax2.bar(hourly[x_col], -hourly["discharge_mw"], color=BRAND["red"], alpha=0.26, label="discharge")
    ax2.set_ylabel(power_label)
    lines, labels = ax1.get_legend_handles_labels()
    bars, bar_labels = ax2.get_legend_handles_labels()
    ax1.legend(lines + bars, labels + bar_labels, loc="upper right", fontsize=8)
    ax1.set_title(title, loc="left")
    ax2.spines["top"].set_visible(False)
    fig.tight_layout()
    _save_static(fig, path)
    plt.close(fig)
    return FigureRecord(str(path), "Storage SOC and hourly charge/discharge profile.")


def _plot_indoor_temperature(indoor: pd.DataFrame, path: Path, scenario: MVPScenario | None) -> FigureRecord:
    fig, ax = plt.subplots(figsize=(11.2, 5.3))
    long_horizon = int(indoor["hour"].max()) > 720
    if long_horizon:
        frame = indoor.copy()
        frame["day"] = frame["hour"] // 24
        for building_id, group in frame.groupby("building_id"):
            daily = group.groupby("day", as_index=False)["indoor_temp_c"].min()
            ax.plot(daily["day"], daily["indoor_temp_c"], linewidth=1.7, label=f"{building_id} daily min")
        xlabel = "Day"
        title = "Daily Minimum Indoor Temperature Verification"
    else:
        for building_id, group in indoor.groupby("building_id"):
            ax.plot(group["hour"], group["indoor_temp_c"], linewidth=1.7, label=str(building_id))
        xlabel = "Hour"
        title = "Indoor Temperature Constraint Verification"
    if scenario is not None:
        mins = [building.comfort_min_c for building in scenario.buildings]
        maxs = [building.comfort_max_c for building in scenario.buildings]
        ax.axhspan(min(mins), max(maxs), color=BRAND["blue_soft"], alpha=0.08, label="comfort band")
        ax.axhline(min(mins), color=BRAND["red"], linestyle="--", linewidth=1.2, label="comfort min")
        ax.axhline(max(maxs), color=BRAND["red"], linestyle=":", linewidth=1.2, label="comfort max")
    ax.set_title(title, loc="left")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Indoor temperature / C")
    _clean_axis(ax)
    ax.legend(loc="center right", fontsize=8, ncols=1)
    fig.tight_layout()
    _save_static(fig, path)
    plt.close(fig)
    return FigureRecord(str(path), "Indoor temperature trajectories against comfort bounds.")


def _plot_capacity(capacity: pd.DataFrame, path: Path) -> FigureRecord:
    frame = capacity.copy()
    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    colors = [SOURCE_COLORS.get(source, BRAND["violet"] if source == "storage" else BRAND["blue_soft"]) for source in frame["source_id"]]
    bars = ax.bar(frame["source_id"], frame["capacity_mw"], color=colors, edgecolor=BRAND["ink"], linewidth=0.5)
    for bar, value in zip(bars, frame["capacity_mw"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.2f}", ha="center", va="bottom", fontsize=8, color=BRAND["ink"])
    ax.set_title("Optimized Capacity Mix", loc="left")
    ax.set_xlabel("Asset")
    ax.set_ylabel("Capacity / MW or MWh")
    ax.tick_params(axis="x", rotation=25)
    _clean_axis(ax, x_grid=False)
    fig.tight_layout()
    _save_static(fig, path)
    plt.close(fig)
    return FigureRecord(str(path), "Optimized heat-source and storage capacity mix.")


def _plot_spatial_network_html(assignment: pd.DataFrame, scenario: MVPScenario, path: Path) -> FigureRecord:
    buildings = {building.building_id: building for building in scenario.buildings}
    sites = {site.site_id: site for site in scenario.candidate_sites}
    assigned = assignment[assignment["assigned"] == 1]
    open_sites = set(assignment[assignment["site_open"] == 1]["site_id"].astype(str))

    fig = go.Figure()
    for row in assigned.itertuples(index=False):
        building = buildings[str(row.building_id)]
        site = sites[str(row.site_id)]
        fig.add_trace(
            go.Scatter(
                x=[site.lon, building.lon],
                y=[site.lat, building.lat],
                mode="lines",
                line={"width": max(1.4, 3.0 - float(row.distance_km) * 0.2), "color": BRAND["muted"]},
                hoverinfo="text",
                text=f"{row.site_id} -> {row.building_id}, {row.distance_km:.2f} km",
                showlegend=False,
            )
        )
    for site in scenario.candidate_sites:
        radius_lon, radius_lat = _radius_circle(site.lon, site.lat, site.max_radius_km)
        fig.add_trace(
            go.Scatter(
                x=radius_lon,
                y=radius_lat,
                mode="lines",
                line={"width": 1, "color": "rgba(15,77,146,0.22)", "dash": "dot"},
                hoverinfo="skip",
                showlegend=False,
            )
        )
    fig.add_trace(
        go.Scatter(
            x=[site.lon for site in scenario.candidate_sites],
            y=[site.lat for site in scenario.candidate_sites],
            mode="markers+text",
            marker={
                "size": [16 if site.site_id in open_sites else 11 for site in scenario.candidate_sites],
                "color": [BRAND["blue"] if site.site_id in open_sites else "#B7C1CE" for site in scenario.candidate_sites],
                "symbol": "square",
                "line": {"width": 1, "color": BRAND["ink"]},
            },
            text=[site.site_id for site in scenario.candidate_sites],
            textposition="top center",
            name="Candidate sites",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[building.lon for building in scenario.buildings],
            y=[building.lat for building in scenario.buildings],
            mode="markers+text",
            marker={
                "size": [max(8, building.floor_area_m2 / 1200) for building in scenario.buildings],
                "color": BRAND["green"],
                "opacity": 0.82,
                "line": {"width": 0.8, "color": BRAND["ink"]},
            },
            text=[building.building_id for building in scenario.buildings],
            textposition="bottom center",
            name="Buildings",
        )
    )
    fig.update_layout(
        title="Spatial Site Assignment and Service Radius",
        xaxis_title="Longitude",
        yaxis_title="Latitude",
        template="plotly_white",
        width=920,
        height=680,
    )
    fig.write_html(path, include_plotlyjs="cdn")
    return FigureRecord(str(path), "Interactive spatial assignment and candidate-site network.")


def _plot_spatial_network_map_html(assignment: pd.DataFrame, scenario: MVPScenario, path: Path) -> FigureRecord:
    import folium
    from branca.element import Element

    buildings = {building.building_id: building for building in scenario.buildings}
    sites = {site.site_id: site for site in scenario.candidate_sites}
    assigned = assignment[assignment["assigned"] == 1].copy()
    open_sites = set(assignment[assignment["site_open"] == 1]["site_id"].astype(str))

    all_lats = [building.lat for building in scenario.buildings] + [site.lat for site in scenario.candidate_sites]
    all_lons = [building.lon for building in scenario.buildings] + [site.lon for site in scenario.candidate_sites]
    center = [sum(all_lats) / len(all_lats), sum(all_lons) / len(all_lons)]
    fmap = folium.Map(
        location=center,
        zoom_start=13,
        tiles="CartoDB positron",
        control_scale=True,
        prefer_canvas=True,
    )

    radius_group = folium.FeatureGroup(name="Service radius", show=True)
    link_group = folium.FeatureGroup(name="Selected network links", show=True)
    site_group = folium.FeatureGroup(name="Candidate energy stations", show=True)
    building_group = folium.FeatureGroup(name="Building load points", show=True)

    for site in scenario.candidate_sites:
        is_open = site.site_id in open_sites
        folium.Circle(
            location=[site.lat, site.lon],
            radius=site.max_radius_km * 1000.0,
            color=BRAND["blue"] if is_open else "#9AA8B8",
            weight=1.25 if is_open else 0.85,
            opacity=0.40 if is_open else 0.22,
            fill=True,
            fill_color=BRAND["blue"] if is_open else "#B7C1CE",
            fill_opacity=0.030 if is_open else 0.012,
            dash_array="8 8" if is_open else "4 8",
            tooltip=f"{site.site_id} service radius: {site.max_radius_km:.1f} km",
        ).add_to(radius_group)

    for row in assigned.itertuples(index=False):
        building = buildings[str(row.building_id)]
        site = sites[str(row.site_id)]
        distance = float(row.distance_km)
        locations = [[site.lat, site.lon], [building.lat, building.lon]]
        tooltip = f"{row.site_id} -> {row.building_id}: {distance:.2f} km"
        folium.PolyLine(
            locations=locations,
            color="#FFFFFF",
            weight=max(5.0, 8.0 - distance),
            opacity=0.82,
            tooltip=tooltip,
        ).add_to(link_group)
        folium.PolyLine(
            locations=locations,
            color="#0B4A8B",
            weight=max(2.4, 4.5 - distance * 0.25),
            opacity=0.90,
            tooltip=tooltip,
        ).add_to(link_group)

    for site in scenario.candidate_sites:
        is_open = site.site_id in open_sites
        popup = (
            f"<b>{site.site_id}</b><br>"
            f"Status: {'selected' if is_open else 'candidate'}<br>"
            f"Service capacity: {site.service_capacity_mw:.2f} MW<br>"
            f"Service radius: {site.max_radius_km:.2f} km"
        )
        folium.Marker(
            location=[site.lat, site.lon],
            icon=folium.DivIcon(
                class_name="cp-site-icon",
                html=(
                    f'<div class="cp-site {"cp-site-open" if is_open else "cp-site-candidate"}">'
                    f'<span class="cp-site-core"></span>'
                    f'<span class="cp-site-label">{site.site_id}</span>'
                    "</div>"
                ),
            ),
            popup=folium.Popup(popup, max_width=280),
            tooltip=f"{site.site_id} ({'selected' if is_open else 'candidate'})",
        ).add_to(site_group)

    for building in scenario.buildings:
        marker_size = max(14.0, min(28.0, building.floor_area_m2 / 1500.0))
        popup = (
            f"<b>{building.building_id}</b><br>"
            f"Floor area: {building.floor_area_m2:,.0f} m2<br>"
            f"Peak low-grade heat: {building.peak_heat_mw:.2f} MW<br>"
            f"Comfort band: {building.comfort_min_c:.1f}-{building.comfort_max_c:.1f} C"
        )
        folium.Marker(
            location=[building.lat, building.lon],
            icon=folium.DivIcon(
                class_name="cp-building-icon",
                html=(
                    f'<div class="cp-building" style="width:{marker_size:.1f}px;height:{marker_size:.1f}px;'
                    f'margin-left:{-marker_size / 2:.1f}px;margin-top:{-marker_size / 2:.1f}px;"></div>'
                ),
            ),
            popup=folium.Popup(popup, max_width=300),
            tooltip=f"{building.building_id}: {building.floor_area_m2:,.0f} m2",
        ).add_to(building_group)

    radius_group.add_to(fmap)
    link_group.add_to(fmap)
    site_group.add_to(fmap)
    building_group.add_to(fmap)
    folium.LayerControl(collapsed=False).add_to(fmap)

    bounds = [[min(all_lats), min(all_lons)], [max(all_lats), max(all_lons)]]
    fmap.fit_bounds(bounds, padding=(42, 42))
    fmap.get_root().html.add_child(Element(_map_legend_html(scenario.scenario_id, len(open_sites), len(assigned))))
    path.parent.mkdir(parents=True, exist_ok=True)
    fmap.save(str(path))
    return FigureRecord(str(path), "Folium map with load points, selected stations, service radii, and active network links.")


def _map_legend_html(scenario_id: str, open_site_count: int, assigned_link_count: int) -> str:
    return f"""
<style>
  .leaflet-control-layers {{
    border: 1px solid #D7DEE8 !important;
    border-radius: 6px !important;
    box-shadow: 0 12px 30px rgba(31,41,51,.12) !important;
    font-family: Arial, sans-serif;
    color: #1F2933;
  }}
  .leaflet-control-layers-expanded {{
    padding: 10px 12px !important;
  }}
  .leaflet-control-zoom {{
    border: 1px solid #D7DEE8 !important;
    border-radius: 6px !important;
    overflow: hidden;
    box-shadow: 0 10px 26px rgba(31,41,51,.10) !important;
  }}
  .leaflet-control-zoom a {{
    color: #1F2933 !important;
    border-bottom-color: #D7DEE8 !important;
  }}
  .cp-site {{
    position: relative;
    transform: translate(-16px, -16px);
    width: 32px;
    height: 32px;
  }}
  .cp-site-core {{
    position: absolute;
    left: 6px;
    top: 6px;
    width: 20px;
    height: 20px;
    border-radius: 6px;
    border: 2px solid #FFFFFF;
    box-shadow: 0 8px 18px rgba(31,41,51,.22);
    background: #9AA8B8;
  }}
  .cp-site-open .cp-site-core {{
    background: #0B4A8B;
    border-color: #FFFFFF;
    box-shadow: 0 0 0 4px rgba(15,77,146,.18), 0 10px 22px rgba(15,77,146,.26);
  }}
  .cp-site-label {{
    position: absolute;
    left: 22px;
    top: -2px;
    white-space: nowrap;
    padding: 3px 7px;
    border-radius: 4px;
    background: rgba(255,255,255,.92);
    border: 1px solid rgba(215,222,232,.95);
    color: #1F2933;
    font-size: 11px;
    font-weight: 700;
    box-shadow: 0 6px 15px rgba(31,41,51,.10);
  }}
  .cp-site-candidate .cp-site-label {{
    color: #6B7280;
    font-weight: 600;
  }}
  .cp-building {{
    border-radius: 999px;
    border: 2px solid #FFFFFF;
    background: #5FA777;
    box-shadow: 0 0 0 3px rgba(95,167,119,.20), 0 8px 18px rgba(31,41,51,.18);
    opacity: .88;
  }}
</style>
<div style="
 position: fixed;
 top: 18px;
 left: 58px;
 z-index: 9999;
 width: 330px;
 background: rgba(255,255,255,0.96);
 border: 1px solid #D7DEE8;
 border-radius: 6px;
 box-shadow: 0 12px 30px rgba(31,41,51,.14);
 padding: 16px 17px;
 font-family: Arial, sans-serif;
 color: #1F2933;">
  <div style="font-size: 12px; color: #0F4D92; font-weight: 700; letter-spacing: .05em; text-transform: uppercase;">CP202611 Spatial Planning</div>
  <div style="font-size: 16px; font-weight: 700; margin-top: 5px; line-height: 1.25;">{scenario_id}</div>
  <div style="display:flex; gap:8px; margin: 11px 0 12px;">
    <div style="flex:1; padding:8px 9px; border:1px solid #E8EDF3; border-radius:5px; background:#F8FAFD;">
      <div style="font-size:18px; font-weight:700; color:#0F4D92;">{open_site_count}</div>
      <div style="font-size:11px; color:#6B7280;">selected stations</div>
    </div>
    <div style="flex:1; padding:8px 9px; border:1px solid #E8EDF3; border-radius:5px; background:#F8FAFD;">
      <div style="font-size:18px; font-weight:700; color:#0F4D92;">{assigned_link_count}</div>
      <div style="font-size:11px; color:#6B7280;">active links</div>
    </div>
  </div>
  <div style="display: grid; gap: 7px; font-size: 12px;">
    <div><span style="display:inline-block;width:12px;height:12px;border-radius:50%;background:#5FA777;border:2px solid #fff;box-shadow:0 0 0 2px rgba(95,167,119,.20);margin-right:8px;vertical-align:-2px;"></span>Building load point, scaled by floor area</div>
    <div><span style="display:inline-block;width:13px;height:13px;border-radius:4px;background:#0B4A8B;border:2px solid #fff;box-shadow:0 0 0 3px rgba(15,77,146,.18);margin-right:8px;vertical-align:-3px;"></span>Selected candidate energy station</div>
    <div><span style="display:inline-block;width:25px;height:4px;background:#0B4A8B;border-top:2px solid #fff;border-bottom:2px solid #fff;margin-right:8px;vertical-align:middle;"></span>Activated station-building service link</div>
    <div><span style="display:inline-block;width:16px;height:16px;border-radius:50%;border:1px dashed #0B4A8B;background:rgba(15,77,146,.04);margin-right:8px;vertical-align:middle;"></span>Maximum service radius</div>
  </div>
</div>
"""


def _plot_energy_sankey_html(dispatch: pd.DataFrame, network_edges: pd.DataFrame, path: Path) -> FigureRecord:
    source_grade = (
        dispatch[(dispatch["heat_mw"] > 1e-9) & (~dispatch["source_id"].isin(["storage_charge"]))]
        .groupby(["source_id", "grade"], as_index=False)["heat_mw"]
        .sum()
    )
    grade_delivery = network_edges.groupby("building_id", as_index=False)[["low_delivered_mw", "mid_delivered_mw"]].sum()
    labels: list[str] = []
    source_idx: list[int] = []
    target_idx: list[int] = []
    values: list[float] = []

    def node(label: str) -> int:
        if label not in labels:
            labels.append(label)
        return labels.index(label)

    grade_nodes = {"low": node("low-temperature network"), "mid": node("mid-temperature network")}
    for row in source_grade.itertuples(index=False):
        value = float(row.heat_mw)
        if value <= 0:
            continue
        source_idx.append(node(str(row.source_id)))
        target_idx.append(grade_nodes[str(row.grade)])
        values.append(value)

    for row in grade_delivery.itertuples(index=False):
        building_node = node(str(row.building_id))
        if float(row.low_delivered_mw) > 0:
            source_idx.append(grade_nodes["low"])
            target_idx.append(building_node)
            values.append(float(row.low_delivered_mw))
        if float(row.mid_delivered_mw) > 0:
            source_idx.append(grade_nodes["mid"])
            target_idx.append(building_node)
            values.append(float(row.mid_delivered_mw))

    sankey_colors = [SOURCE_COLORS.get(label, BRAND["blue_soft"]) if label not in {"low-temperature network", "mid-temperature network"} else (BRAND["teal"] if label.startswith("low") else BRAND["amber"]) for label in labels]
    fig = go.Figure(
        data=[
            go.Sankey(
                node={"label": labels, "pad": 16, "thickness": 16, "color": sankey_colors},
                link={"source": source_idx, "target": target_idx, "value": values, "color": "rgba(15,77,146,0.18)"},
            )
        ]
    )
    fig.update_layout(title="Aggregated Source-Network-Load Energy Flow", template="plotly_white", width=980, height=640)
    fig.write_html(path, include_plotlyjs="cdn")
    return FigureRecord(str(path), "Interactive Sankey diagram of aggregated source-network-load heat flows.")


def _plot_pareto(pareto: pd.DataFrame, path: Path) -> FigureRecord:
    fig, ax = plt.subplots(figsize=(8.5, 5.8))
    scatter = ax.scatter(
        pareto["carbon_emissions_t"],
        pareto["economic_cost_cny"] / 1_000_000.0,
        c=pareto["exergy_loss_mwh_eq"],
        s=80,
        cmap="cividis_r",
        edgecolors=BRAND["ink"],
        linewidths=0.5,
    )
    for row in pareto.itertuples(index=False):
        ax.annotate(str(row.run_id), (row.carbon_emissions_t, row.economic_cost_cny / 1_000_000.0), xytext=(4, 4), textcoords="offset points", fontsize=8)
    ax.set_title("Cost-Carbon-Exergy Pareto Candidates", loc="left")
    ax.set_xlabel("Carbon emissions / t")
    ax.set_ylabel("Economic cost / million CNY")
    _clean_axis(ax)
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Exergy loss / MWh-eq")
    fig.tight_layout()
    _save_static(fig, path)
    plt.close(fig)
    return FigureRecord(str(path), "Cost-carbon-exergy Pareto candidate scatter plot.")


def _clean_axis(ax, x_grid: bool = True) -> None:
    ax.grid(axis="y", color=BRAND["grid"], linewidth=0.8, alpha=0.9)
    if x_grid:
        ax.grid(axis="x", color=BRAND["grid"], linewidth=0.5, alpha=0.55)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.tick_params(length=3, width=0.8)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))


def _save_static(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=240, bbox_inches="tight")
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight")


def _radius_circle(lon: float, lat: float, radius_km: float, points: int = 120) -> tuple[list[float], list[float]]:
    import math

    lon_scale = 111.320 * math.cos(math.radians(lat))
    lat_scale = 110.574
    xs = []
    ys = []
    for i in range(points + 1):
        theta = 2.0 * math.pi * i / points
        xs.append(lon + radius_km * math.cos(theta) / lon_scale)
        ys.append(lat + radius_km * math.sin(theta) / lat_scale)
    return xs, ys


def _write_visual_dashboard(output_dir: Path, figures: list[FigureRecord]) -> None:
    image_cards = []
    html_cards = []
    for record in figures:
        file_path = Path(record.file)
        rel = file_path.name
        if file_path.suffix.lower() == ".html":
            html_cards.append(f'<section><h2>{rel}</h2><iframe src="{rel}" loading="lazy"></iframe><p>{record.description}</p></section>')
        else:
            image_cards.append(f'<section><h2>{rel}</h2><img src="{rel}" alt="{record.description}"><p>{record.description}</p></section>')
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>CP202611 Visual Report</title>
  <style>
    :root {{
      --ink: {BRAND["ink"]};
      --muted: {BRAND["muted"]};
      --line: {BRAND["hairline"]};
      --blue: {BRAND["blue"]};
      --surface: #ffffff;
      --soft: #f5f7fb;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Arial, "DejaVu Sans", sans-serif;
      color: var(--ink);
      background: linear-gradient(180deg, #f7f9fc 0%, #ffffff 26%);
    }}
    header {{
      padding: 44px min(6vw, 72px) 28px;
      border-bottom: 1px solid var(--line);
    }}
    .eyebrow {{
      color: var(--blue);
      font-size: 13px;
      letter-spacing: .08em;
      text-transform: uppercase;
      font-weight: 700;
    }}
    h1 {{
      max-width: 920px;
      margin: 10px 0 12px;
      font-size: clamp(30px, 4vw, 56px);
      line-height: 1.04;
      letter-spacing: 0;
    }}
    header p {{
      max-width: 760px;
      color: var(--muted);
      font-size: 16px;
      line-height: 1.65;
    }}
    main {{
      padding: 26px min(6vw, 72px) 60px;
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 26px;
    }}
    section {{
      background: var(--surface);
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 18px;
      box-shadow: 0 14px 36px rgba(31,41,51,.06);
    }}
    section h2 {{
      margin: 0 0 12px;
      font-size: 14px;
      letter-spacing: 0;
    }}
    section p {{
      margin: 12px 0 0;
      color: var(--muted);
      font-size: 13px;
      line-height: 1.5;
    }}
    img {{
      width: 100%;
      display: block;
      border: 1px solid #edf1f6;
      border-radius: 4px;
      background: white;
    }}
    iframe {{
      width: 100%;
      height: 520px;
      border: 1px solid #edf1f6;
      border-radius: 4px;
      background: white;
    }}
    @media (max-width: 900px) {{
      main {{ grid-template-columns: 1fr; }}
      header {{ padding-top: 30px; }}
    }}
  </style>
</head>
<body>
  <header>
    <div class="eyebrow">CP202611 source-network-load-storage-quality model</div>
    <h1>Visual evidence for clean district heating planning.</h1>
    <p>Automatically generated figures from the optimization outputs. The pack is intended for report drafting, slide review, and quick inspection before official data are connected.</p>
  </header>
  <main>
    {''.join(image_cards)}
    {''.join(html_cards)}
  </main>
</body>
</html>"""
    (output_dir / "report_dashboard.html").write_text(html, encoding="utf-8")
