# 官方数据接入操作手册

目标：官方数据到达后，在不改优化模型的前提下完成接入、校验、求解和图表生成。

## 1. 接入原则

1. 官方原始文件放入 `data/raw/official/`。
2. 不直接修改官方原始文件。
3. 只修改 `configs/field_mapping_official_template.yaml` 或复制出的项目映射文件。
4. 所有原始字段通过适配层转为 `data/processed/official/` 标准数据。
5. 优化、压力测试、Pareto 和可视化只读取 `data/processed/official/`。

## 2. 推荐流程

### Step 1: 复制映射模板

```bash
copy configs\field_mapping_official_template.yaml configs\field_mapping_official.yaml
```

### Step 2: 对照官方表头修改映射

在 `configs/field_mapping_official.yaml` 中修改：

- `file`
- `sheet_name`
- `columns`
- `defaults`
- `scenario`
- `storage`

如果官方同一个字段可能有多个表头，可以写成候选列表：

```yaml
time_series:
  file: official_time.xlsx
  sheet_name: Sheet1
  columns:
    hour: ["hour", "时段", "小时"]
    outdoor_temperature_c: ["outdoor_temperature_c", "室外温度", "干球温度"]
```

适配层会按顺序匹配候选表头，并自动去除表头首尾空格和 BOM。

### Step 3: 预处理官方数据

```bash
python -m cp202611.cli preprocess-official ^
  --raw-dir data\raw\official ^
  --mapping configs\field_mapping_official.yaml ^
  --output-dir data\processed\official
```

检查输出：

```text
data/processed/official/
  scenario.yaml
  time_series.csv
  buildings.csv
  building_mid_temp_demand.csv
  heat_sources.csv
  candidate_sites.csv
  storage.yaml
  validation_issues.csv
  validation_report.md
```

### Step 4: 运行一键验收

```bash
python -m cp202611.cli acceptance-check ^
  --dataset-dir data\processed\official ^
  --output-dir outputs\acceptance_official ^
  --n-days 8 ^
  --max-iterations 4 ^
  --stress-mode full
```

快速冒烟检查可先使用：

```bash
python -m cp202611.cli acceptance-check ^
  --dataset-dir data\processed\official ^
  --output-dir outputs\acceptance_official_smoke ^
  --n-days 4 ^
  --max-iterations 2 ^
  --carbon-prices 0 ^
  --exergy-penalties 0 ^
  --stress-mode quick ^
  --skip-visuals
```

核心输出：

```text
outputs/acceptance_official/
  acceptance_summary.csv
  acceptance_report.md
  validation/
  benchmark/
  stress/
  pareto/
  figures/
```

### Step 5: 运行鲁棒性矩阵

官方数据通过基础验收后，再运行多扰动鲁棒性矩阵。该步骤用于比较不同外部边界下的成本、碳排、㶲损、舒适度和欠热指标，判断方案是否只对单一数据集成立。

```bash
python -m cp202611.cli robustness-matrix ^
  --dataset-dir data\processed\official ^
  --output-dir outputs\robustness_official ^
  --n-days 4 ^
  --max-iterations 3 ^
  --mode full
```

如果只是检查官方数据能否跑通，可先用快速模式：

```bash
python -m cp202611.cli robustness-matrix ^
  --dataset-dir data\processed\official ^
  --output-dir outputs\robustness_official_smoke ^
  --n-days 3 ^
  --max-iterations 1 ^
  --mode quick
```

### Step 6: 生成交叉审阅证据包

官方数据完成基础验收与鲁棒性矩阵后，生成一份可直接发给组员或外部 AI 的证据包。该步骤不重新求解模型，只汇总已有结果。

```bash
python -m cp202611.cli evidence-package ^
  --acceptance-dir outputs\acceptance_official ^
  --robustness-dir outputs\robustness_official ^
  --benchmark-dir outputs\acceptance_official\benchmark ^
  --dataset-label official ^
  --output-dir outputs\evidence_official
```

## 3. 必看验收项

官方数据接入后，必须检查：

| 文件 | 检查内容 |
|---|---|
| `validation/validation_report.md` | 是否存在错误、单位异常、不可达站点 |
| `benchmark/benchmark_summary.csv` | 是否收敛，是否 1h 全季，是否无欠供 |
| `stress/stress_results.csv` | 压力情景是否全部收敛 |
| `robustness_matrix.csv` | 多扰动情景下成本、碳排、㶲损和舒适性是否仍然通过 |
| `pareto/pareto_runs.csv` | 成本、碳排、㶲损扫描是否方向合理 |
| `figures/spatial_network_map.html` | 坐标、服务半径、管线是否合理 |
| `acceptance_report.md` | 总体验收是否通过 |
| `evidence_package.md` | 评分点证据、不可过度宣称内容、外部审阅材料是否完整 |

## 4. 常见问题与处理

### 4.1 表头不一致

现象：`raw column ... not found`

处理：在映射文件中把该字段写成候选列表，或核对 Excel sheet 名称。

### 4.2 小时乱序

适配层会自动排序，不需要手工修改原始文件。

### 4.3 小时重复

现象：`duplicate hour`

处理：检查官方数据是否有重复导出、重复日期或多个区域混在同一时间表。

### 4.4 建筑逐时需求缺失

现象：`mid-temperature demand is missing hours`

处理：补齐缺失小时。若官方确实没有中温需求，可在报告中说明并以 0 或标准曲线填充，但必须记录填充规则。

### 4.5 经纬度不是实际地理坐标

若官方给的是脱敏坐标或平面坐标：

- 地图图层只能作为空间拓扑代理。
- 报告中写明“坐标已脱敏，不代表真实街区位置”。
- 不要使用真实街道名做过度解释。

### 4.6 模型无解

按顺序排查：

1. `validation_report.md` 是否有错误。
2. 中温热源容量是否低于中温峰值需求。
3. 是否有建筑不在任何站点服务半径内。
4. 电网接入上限是否过低。
5. 储热 SOC 周期闭合是否与数据时长匹配。

## 5. 官方数据到达后的交叉审阅材料

若需要给外部 AI 或组员交叉审阅，建议发送：

- `configs/field_mapping_official.yaml`
- `data/processed/official/validation_report.md`
- `outputs/acceptance_official/acceptance_report.md`
- `outputs/acceptance_official/benchmark/benchmark_summary.csv`
- `outputs/acceptance_official/stress/stress_results.csv`
- `outputs/acceptance_official/pareto/pareto_runs.csv`
- `outputs/acceptance_official/figures/spatial_network_map.html` 截图
