# CP202611 代码工程

本目录只放代码、可运行配置、输入数据和输出结果。当前阶段目标是跑通“强力核心 MVP”并形成可替换官方数据的标准输入契约。

## 已实现范围

- 北方中小城市清洁替代场景。
- 双温度等级：低温空间供暖、中温热负荷。
- 源-网-荷-储-质协同：空气源热泵、燃气锅炉、生物质锅炉、电锅炉、储热水箱、线性管网边流、等效㶲损惩罚。
- 多尺度耦合：城市级电网价格/碳因子/接入上限，区域级热微网平衡，用户级 RC 室温动态。
- 空间布局：候选能源站、服务半径、站点容量、管网绕行距离折算、长度投资和热损。
- 舒适性控制：硬性室温上下限叠加目标温度软惩罚，避免优化结果长期贴住最低舒适边界。
- 双层规划：峰值保持的多特征加权 K-medoids 典型日容量规划，全供暖季固定容量验证。
- 反馈修正：若全季验证出现欠热或室温违约，自动加入最差日期重新规划。
- 数据质量诊断、压力测试、Pareto 扫描、官方数据适配模板、pytest 测试。

当前合成数据用于验证模型逻辑，不代表最终官方脱敏数据。

## 环境

```powershell
D:\anaconda\envs\CP202611\python.exe -m pip install -e .[test]
D:\anaconda\envs\CP202611\python.exe -m pytest
```

## 常用命令

```powershell
# 强力核心 MVP
D:\anaconda\envs\CP202611\python.exe -m cp202611.cli mvp --output outputs\mvp_dispatch.csv

# 生成合成周数据并校验
D:\anaconda\envs\CP202611\python.exe -m cp202611.cli generate-synthetic --output-dir data\processed\synthetic_week --hours 168
D:\anaconda\envs\CP202611\python.exe -m cp202611.cli validate-data --dataset-dir data\processed\synthetic_week --output-dir outputs\validation

# 全周直接求解与典型日选择
D:\anaconda\envs\CP202611\python.exe -m cp202611.cli solve --dataset-dir data\processed\synthetic_week --output outputs\week_dispatch.csv
D:\anaconda\envs\CP202611\python.exe -m cp202611.cli typical-days --dataset-dir data\processed\synthetic_week --n-days 4 --metrics-output outputs\typical_day_metrics.csv

# 双层规划、反馈修正、Pareto、压力测试和鲁棒性矩阵
D:\anaconda\envs\CP202611\python.exe -m cp202611.cli two-stage --dataset-dir data\processed\synthetic_week --n-days 4 --output-dir outputs\two_stage
D:\anaconda\envs\CP202611\python.exe -m cp202611.cli feedback-plan --dataset-dir data\processed\synthetic_week --n-days 4 --max-iterations 4 --output-dir outputs\feedback_plan
D:\anaconda\envs\CP202611\python.exe -m cp202611.cli pareto-scan --dataset-dir data\processed\synthetic_week --carbon-prices 0,500,1500 --exergy-penalties 0,260 --n-days 4 --max-iterations 4 --output-dir outputs\pareto
D:\anaconda\envs\CP202611\python.exe -m cp202611.cli stress-test --dataset-dir data\processed\synthetic_week --n-days 4 --max-iterations 3 --output-dir outputs\stress
D:\anaconda\envs\CP202611\python.exe -m cp202611.cli robustness-matrix --dataset-dir data\processed\synthetic_week --n-days 4 --max-iterations 3 --mode full --output-dir outputs\robustness_matrix
D:\anaconda\envs\CP202611\python.exe -m cp202611.cli visualize-results --result-dir outputs\feedback_plan --dataset-dir data\processed\synthetic_week --pareto-dir outputs\pareto --output-dir outputs\figures

# 2880h 合成供暖季端到端 benchmark
D:\anaconda\envs\CP202611\python.exe -m cp202611.cli generate-synthetic --profile season --output-dir data\processed\synthetic_season --hours 2880
D:\anaconda\envs\CP202611\python.exe -m cp202611.cli validate-data --dataset-dir data\processed\synthetic_season --output-dir outputs\validation_season
D:\anaconda\envs\CP202611\python.exe -m cp202611.cli benchmark-plan --dataset-dir data\processed\synthetic_season --n-days 8 --max-iterations 4 --output-dir outputs\benchmark_season
D:\anaconda\envs\CP202611\python.exe -m cp202611.cli robustness-matrix --dataset-dir data\processed\synthetic_season --n-days 4 --max-iterations 3 --mode full --output-dir outputs\robustness_season

# 阶段证据包，用于组员/外部 AI 交叉审阅
D:\anaconda\envs\CP202611\python.exe -m cp202611.cli evidence-package --acceptance-dir outputs\acceptance_smoke --robustness-dir outputs\robustness_week --benchmark-dir outputs\benchmark_season --dataset-label synthetic_pre_official --output-dir outputs\evidence_package
```

## 官方数据适配

官方脱敏数据到位后，不直接改优化模型。先把原始表放入 `data/raw/official/`，然后修改 `configs/field_mapping_official_template.yaml` 中的文件名和原始列名，运行：

```powershell
D:\anaconda\envs\CP202611\python.exe -m cp202611.cli preprocess-official --raw-dir data\raw\official --mapping configs\field_mapping_official_template.yaml --output-dir data\processed\official
D:\anaconda\envs\CP202611\python.exe -m cp202611.cli validate-data --dataset-dir data\processed\official --output-dir outputs\validation_official
D:\anaconda\envs\CP202611\python.exe -m cp202611.cli robustness-matrix --dataset-dir data\processed\official --n-days 4 --max-iterations 3 --mode quick --output-dir outputs\robustness_official_smoke
D:\anaconda\envs\CP202611\python.exe -m cp202611.cli evidence-package --acceptance-dir outputs\acceptance_official_smoke --robustness-dir outputs\robustness_official_smoke --dataset-label official_smoke --output-dir outputs\evidence_official_smoke
```

适配层会输出标准化数据集和 `validation_report.md`。如果官方字段、单位或缺失项与预期不一致，应优先修改映射表或预处理规则，不应绕过 `data/processed/` 直接接入 Pyomo 模型。

`robustness-matrix` 用于比较 base、严寒、限电、COP 下降等扰动下的成本、碳排、㶲损、舒适度和欠热指标。它不是替代 `acceptance-check`，而是用于证明模型没有只对单一合成数据“调参过拟合”；官方数据到来后，同一命令可直接用于官方标准化数据集。

`evidence-package` 不重新求解模型，只汇总已有 `acceptance`、`robustness` 和可选 `benchmark` 输出，生成可交叉审阅的 `evidence_package.md`。该文件适合发给组员或外部 AI 审阅，重点说明当前证据、评分点对应关系和不可过度宣称的边界。

## 可视化输出

`visualize-results` 会从既有求解结果生成报告图，不重新求解模型。静态图同时输出 PNG 和可编辑 SVG，交互图输出 HTML，并生成 `report_dashboard.html` 作为轻量可视化总览页。该页面不是官方硬性要求的前端系统，而是便于汇报、答辩和成果展示的辅助交付物。
