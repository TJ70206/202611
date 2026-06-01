# CP202611 标准数据契约

本契约定义优化模型读取的标准输入。官方原始数据、公用数据和合成数据都必须先经过适配层转换为本契约，再进入 Pyomo 优化模型。

核心原则：

- 优化模型只读取 `data/processed/<scenario>/`。
- 原始数据不直接进入优化模型。
- 所有功率统一为 MW，能量统一为 MWh，价格统一为 CNY/MWh 或 CNY/MW。
- 经纬度字段若为脱敏坐标，报告中必须说明为规划拓扑代理，不得声称是真实街区位置。

## 1. 目录结构

```text
data/processed/<scenario>/
  scenario.yaml
  time_series.csv
  buildings.csv
  building_mid_temp_demand.csv
  heat_sources.csv
  candidate_sites.csv
  storage.yaml
```

## 2. `scenario.yaml`

必需字段：

| 字段 | 单位 | 含义 |
|---|---:|---|
| `scenario_id` | - | 场景编号 |
| `electricity_base_price_cny_per_mwh` | CNY/MWh_e | 基准电价 |
| `gas_price_cny_per_mwh_fuel` | CNY/MWh_fuel | 天然气燃料价格 |
| `biomass_price_cny_per_mwh_fuel` | CNY/MWh_fuel | 生物质燃料价格 |
| `gas_emission_factor_t_per_mwh_fuel` | tCO2e/MWh_fuel | 天然气排放因子 |
| `biomass_emission_factor_t_per_mwh_fuel` | tCO2e/MWh_fuel | 生物质排放因子 |
| `carbon_price_cny_per_t` | CNY/tCO2e | 碳价格 |
| `capex_recovery_factor` | 1/year | 投资年化系数 |
| `operation_weight` | - | 代表时段运行权重 |
| `exergy_penalty_cny_per_mwh` | CNY/MWh-eq | 等效㶲损惩罚 |
| `comfort_slack_penalty_cny_per_c_h` | CNY/(C h) | 室温违约惩罚 |
| `unmet_heat_penalty_cny_per_mwh` | CNY/MWh | 欠供惩罚 |
| `max_open_sites` | count | 最多启用站点数 |
| `pipe_loss_fraction_per_km` | 1/km | 管网线性热损系数 |
| `pipe_capex_cny_per_mw_km` | CNY/(MW km) | 管网单位容量长度投资 |
| `pipe_fixed_om_fraction` | 1/year | 管网固定运维比例 |
| `pipe_capacity_margin` | - | 管网容量裕度 |
| `storage_cycle_block_size_h` | h 或 null | 储热周期闭合块 |

## 3. `time_series.csv`

| 字段 | 单位 | 含义 |
|---|---:|---|
| `hour` | h | 从 0 开始的小时序号 |
| `outdoor_temperature_c` | C | 室外温度 |
| `electricity_price_multiplier` | - | 分时电价倍率 |
| `grid_carbon_factor_t_per_mwh` | tCO2e/MWh_e | 电网碳因子 |
| `grid_import_limit_mw` | MW_e | 城市级电网接入上限 |
| `time_weight` | - | 可选，代表时段权重 |

验收规则：

- `hour` 不允许重复。
- 小时序列允许乱序，但适配层会排序。
- 时间长度必须为完整小时序列，最终验证场景应覆盖完整供暖季。

## 4. `buildings.csv`

| 字段 | 单位 | 含义 |
|---|---:|---|
| `building_id` | - | 建筑编号 |
| `lon` | degree 或脱敏坐标 | 经度或平面代理横坐标 |
| `lat` | degree 或脱敏坐标 | 纬度或平面代理纵坐标 |
| `floor_area_m2` | m2 | 建筑面积 |
| `heat_loss_mw_per_c` | MW/C | 一阶 RC 热损系数 |
| `thermal_capacity_mwh_per_c` | MWh/C | 建筑热容 |
| `initial_indoor_temp_c` | C | 初始室温 |
| `comfort_min_c` | C | 舒适下限 |
| `comfort_max_c` | C | 舒适上限 |
| `internal_gain_mw` | MW | 内部得热 |
| `peak_heat_mw` | MW | 空间供暖峰值约束 |

## 5. `building_mid_temp_demand.csv`

| 字段 | 单位 | 含义 |
|---|---:|---|
| `building_id` | - | 建筑编号 |
| `hour` | h | 小时序号 |
| `mid_temp_demand_mw` | MW | 中温热需求 |

验收规则：

- 每个建筑必须覆盖 `time_series.csv` 的全部小时。
- 建筑和小时组合不允许重复。
- 缺小时会在官方适配阶段直接报错。

## 6. `heat_sources.csv`

| 字段 | 单位 | 含义 |
|---|---:|---|
| `source_id` | - | 热源编号 |
| `fuel` | - | `electricity`, `gas`, `biomass` |
| `allowed_grades` | - | 可供温度等级，使用 `low|mid` |
| `max_capacity_mw` | MW_th | 最大可建容量 |
| `capex_cny_per_mw` | CNY/MW_th | 单位投资 |
| `fixed_om_fraction` | 1/year | 固定运维比例 |
| `efficiency` | - | 热效率，电热泵填 1 并用 `base_cop` |
| `base_cop` | - | 热泵基准 COP，非热泵可为空 |
| `variable_om_cny_per_mwh_th` | CNY/MWh_th | 可变运维 |
| `exergy_loss_coeff_by_grade` | - | 各温度等级等效㶲损系数 JSON |

## 7. `candidate_sites.csv`

| 字段 | 单位 | 含义 |
|---|---:|---|
| `site_id` | - | 候选能源站编号 |
| `lon` | degree 或脱敏坐标 | 经度或平面代理横坐标 |
| `lat` | degree 或脱敏坐标 | 纬度或平面代理纵坐标 |
| `max_radius_km` | km | 最大服务半径 |
| `service_capacity_mw` | MW | 站点服务容量 |
| `fixed_cost_cny` | CNY | 启用固定成本 |

## 8. `storage.yaml`

| 字段 | 单位 | 含义 |
|---|---:|---|
| `max_capacity_mwh` | MWh | 最大储热容量 |
| `capex_cny_per_mwh` | CNY/MWh | 储热投资 |
| `fixed_om_fraction` | 1/year | 固定运维 |
| `charge_efficiency` | - | 充热效率 |
| `discharge_efficiency` | - | 放热效率 |
| `standing_loss_fraction_per_h` | 1/h | 静置热损 |
| `power_to_energy_ratio` | 1/h | 功率容量比 |

## 9. 官方数据缺字段处理

若官方数据缺少以下字段，需要用公开资料或标准参数补齐，并在 `validation_report.md` 和报告附录中说明来源：

- 设备投资与运维参数。
- 热源效率或 COP。
- 电价、气价、生物质燃料价格。
- 电力碳因子。
- 建筑热损和热容参数。
- 储热投资、效率和热损。
