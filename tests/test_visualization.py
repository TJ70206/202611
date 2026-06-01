from __future__ import annotations

from cp202611.dataio import write_processed_dataset
from cp202611.visualization import create_report_figures


def _write_result_files(result, result_dir):
    result_dir.mkdir(parents=True, exist_ok=True)
    result.dispatch.to_csv(result_dir / "verification_dispatch.csv", index=False, encoding="utf-8-sig")
    result.storage.to_csv(result_dir / "verification_storage.csv", index=False, encoding="utf-8-sig")
    result.indoor_temperature.to_csv(result_dir / "verification_indoor_temperature.csv", index=False, encoding="utf-8-sig")
    result.network_edges.to_csv(result_dir / "verification_network_edges.csv", index=False, encoding="utf-8-sig")
    result.source_capacity.to_csv(result_dir / "verification_capacity.csv", index=False, encoding="utf-8-sig")
    result.spatial_assignment.to_csv(result_dir / "planning_spatial_assignment.csv", index=False, encoding="utf-8-sig")


def test_create_report_figures_writes_core_outputs(tmp_path, synthetic_scenario, solved_mvp):
    dataset_dir = tmp_path / "dataset"
    result_dir = tmp_path / "results"
    output_dir = tmp_path / "figures"
    write_processed_dataset(synthetic_scenario, dataset_dir)
    _write_result_files(solved_mvp, result_dir)

    result = create_report_figures(result_dir=result_dir, dataset_dir=dataset_dir, output_dir=output_dir)
    files = {path.name for path in output_dir.iterdir()}

    assert len(result.figures) >= 6
    assert "source_dispatch_stack.png" in files
    assert "source_dispatch_stack.svg" in files
    assert "storage_soc_dispatch.png" in files
    assert "indoor_temperature_constraints.png" in files
    assert "capacity_mix.png" in files
    assert "spatial_network_map.html" in files
    assert "spatial_network.html" in files
    assert "energy_sankey.html" in files
    assert "report_dashboard.html" in files
    assert "visualization_manifest.csv" in files

    map_html = (output_dir / "spatial_network_map.html").read_text(encoding="utf-8")
    assert "CP202611 Spatial Planning" in map_html
    assert synthetic_scenario.buildings[0].building_id in map_html
    assert synthetic_scenario.candidate_sites[0].site_id in map_html
