from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class EvidencePackageResult:
    report_path: Path
    overall_ready: bool
    acceptance_checks: int
    robustness_cases: int


def build_evidence_package(
    output_dir: Path,
    acceptance_dir: Path,
    robustness_dir: Path,
    dataset_label: str,
    benchmark_dir: Path | None = None,
) -> EvidencePackageResult:
    """Build a compact cross-review package from existing validation outputs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    acceptance = _read_optional_csv(acceptance_dir / "acceptance_summary.csv")
    robustness = _read_optional_csv(robustness_dir / "robustness_matrix.csv")
    benchmark = _read_optional_csv((benchmark_dir / "benchmark_summary.csv") if benchmark_dir is not None else None)

    acceptance_ready = _all_passed(acceptance, "passed")
    robustness_ready = _all_passed(robustness, "passed")
    overall_ready = bool(acceptance_ready and robustness_ready)

    report = _render_report(
        dataset_label=dataset_label,
        acceptance_dir=acceptance_dir,
        robustness_dir=robustness_dir,
        benchmark_dir=benchmark_dir,
        acceptance=acceptance,
        robustness=robustness,
        benchmark=benchmark,
        overall_ready=overall_ready,
    )
    report_path = output_dir / "evidence_package.md"
    report_path.write_text(report, encoding="utf-8")
    return EvidencePackageResult(
        report_path=report_path,
        overall_ready=overall_ready,
        acceptance_checks=0 if acceptance is None else len(acceptance),
        robustness_cases=0 if robustness is None else len(robustness),
    )


def _read_optional_csv(path: Path | None) -> pd.DataFrame | None:
    if path is None or not path.exists():
        return None
    return pd.read_csv(path)


def _all_passed(frame: pd.DataFrame | None, column: str) -> bool:
    if frame is None or frame.empty or column not in frame.columns:
        return False
    return bool(frame[column].fillna(False).astype(bool).all())


def _render_report(
    dataset_label: str,
    acceptance_dir: Path,
    robustness_dir: Path,
    benchmark_dir: Path | None,
    acceptance: pd.DataFrame | None,
    robustness: pd.DataFrame | None,
    benchmark: pd.DataFrame | None,
    overall_ready: bool,
) -> str:
    lines = [
        "# CP202611 阶段证据包",
        "",
        f"- 数据集：`{dataset_label}`",
        f"- 阶段结论：`{'通过当前合成数据验收' if overall_ready else '存在未通过或缺失证据项'}`",
        "- 重要边界：当前结论只证明模型框架和算法机制可运行，不替代官方脱敏数据的最终结论。",
        "",
        "## 评分点对齐",
        "",
        "| 官方评分维度 | 当前证据 | 证据文件 | 状态 |",
        "|---|---|---|---|",
        "| 模型完整性与逻辑闭环 | 源-网-荷-储-质模型、1h 时序、舒适度/欠热/平衡校验 | acceptance_summary.csv | " + _status(acceptance, "passed") + " |",
        "| 算法寻优能力 | 典型日规划、全季验证、反馈修正、Pareto/压力扰动接口 | acceptance/pareto 与 benchmark 输出 | " + _status(acceptance, "passed") + " |",
        "| 多尺度耦合 | 城市电网边界、区域微网、用户 RC 热惰性均进入模型 | 代码与验收报告 | 已实现待官方数据复核 |",
        "| 空间与能质协同 | 站点半径/管网距离/温度等级/㶲损惩罚 | 求解结果与地图输出 | 已实现待官方坐标复核 |",
        "| 可视化与分析深度 | 地图、Sankey、调度堆叠、SOC、Pareto 图 | figures/ 与 HTML dashboard | 已实现待官方数据重绘 |",
        "| 代码质量与可移植性 | pytest、CLI、YAML/CSV 数据契约、官方适配层 | tests/ 与 docs/ | 已实现 |",
        "",
        "## 验收摘要",
        "",
        f"- 验收目录：`{acceptance_dir}`",
    ]
    lines.extend(_acceptance_lines(acceptance))
    lines.extend(
        [
            "",
            "## 鲁棒性矩阵摘要",
            "",
            f"- 鲁棒性目录：`{robustness_dir}`",
        ]
    )
    lines.extend(_robustness_lines(robustness))
    lines.extend(["", "## Benchmark 摘要", ""])
    if benchmark_dir is not None:
        lines.append(f"- Benchmark 目录：`{benchmark_dir}`")
    lines.extend(_benchmark_lines(benchmark))
    lines.extend(
        [
            "",
            "## 当前不可过度宣称的内容",
            "",
            "- 合成数据只能用于验证模型结构、代码稳定性和算法响应方向，不能作为最终规划结论。",
            "- 合成坐标使用济南附近经纬度作为地图锚点，仅用于展示空间算法效果；官方脱敏坐标到来后必须重绘地图并重新解释。",
            "- 分时电价、燃气价、生物质价格、设备造价和碳因子虽已有资料依据，但仍需在官方数据到来后按比赛口径复核。",
            "- 若官方数据为脱敏平面坐标，地图底图只能作为空间拓扑代理，不应解释为真实街区。",
            "",
            "## 外部 AI / 组员交叉审阅建议",
            "",
            "建议发送以下材料：",
            "",
            "- 本文件 `evidence_package.md`",
            "- `acceptance_summary.csv` 与 `acceptance_report.md`",
            "- `robustness_matrix.csv` 与 `robustness_report.md`",
            "- `benchmark_summary.csv`",
            "- `spatial_network_map.html` 截图",
            "- `CP202611_最终敲定方案.md` 中模型章节",
        ]
    )
    return "\n".join(lines) + "\n"


def _acceptance_lines(acceptance: pd.DataFrame | None) -> list[str]:
    if acceptance is None or acceptance.empty:
        return ["- 未找到验收摘要。"]
    total = len(acceptance)
    passed = int(acceptance["passed"].fillna(False).astype(bool).sum()) if "passed" in acceptance else 0
    lines = [f"- 通过项：`{passed}/{total}`"]
    for row in acceptance.to_dict(orient="records"):
        lines.append(f"- `{row.get('check')}`：`{row.get('passed')}`，值 `{row.get('value')}`，阈值 `{row.get('threshold')}`")
    return lines


def _robustness_lines(robustness: pd.DataFrame | None) -> list[str]:
    if robustness is None or robustness.empty:
        return ["- 未找到鲁棒性矩阵。"]
    total = len(robustness)
    passed = int(robustness["passed"].fillna(False).astype(bool).sum()) if "passed" in robustness else 0
    lines = [f"- 通过场景：`{passed}/{total}`"]
    display_columns = [
        "case_id",
        "passed",
        "cost_delta_pct",
        "carbon_delta_pct",
        "exergy_delta_pct",
        "max_comfort_slack_c",
        "total_unmet_mid_mwh",
    ]
    for row in robustness.to_dict(orient="records"):
        values = {key: row.get(key) for key in display_columns}
        lines.append(f"- `{values['case_id']}`：通过 `{values['passed']}`，成本变化 `{_fmt(values['cost_delta_pct'])}%`，碳排变化 `{_fmt(values['carbon_delta_pct'])}%`，㶲损变化 `{_fmt(values['exergy_delta_pct'])}%`，舒适松弛 `{_fmt(values['max_comfort_slack_c'], scientific=True)}`。")
    return lines


def _benchmark_lines(benchmark: pd.DataFrame | None) -> list[str]:
    if benchmark is None or benchmark.empty:
        return ["- 未找到 Benchmark 摘要。"]
    row = benchmark.iloc[0].to_dict()
    keys = [
        "horizon_hours",
        "typical_day_count",
        "converged",
        "feedback_iterations",
        "elapsed_seconds",
        "max_comfort_slack_c",
        "total_unmet_mid_mwh",
        "max_low_balance_abs_mw",
        "max_mid_balance_abs_mw",
    ]
    return [f"- `{key}`：`{row.get(key)}`" for key in keys if key in row]


def _status(frame: pd.DataFrame | None, column: str) -> str:
    if frame is None or frame.empty:
        return "缺证据"
    return "通过" if _all_passed(frame, column) else "需复核"


def _fmt(value: object, scientific: bool = False) -> str:
    if value is None or pd.isna(value):
        return ""
    number = float(value)
    if scientific:
        return f"{number:.3e}"
    return f"{number:.2f}"
