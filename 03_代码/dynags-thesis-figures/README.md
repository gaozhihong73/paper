# DynaGS thesis figures

本项目用于生成 DynaGS 硕士学位论文中的科研图。项目通过 `uv` 管理独立 Python 环境，不依赖系统 Python 中已有的软件包。

## 数据与输出

- 原始数据目录：`../../04_数据/真实玉米数据集`
- 正式图件目录：`../../07_论文图表`
- 临时输出目录：`outputs`

绘图脚本不得修改原始数据。正式图件默认只输出 300 dpi PNG；图中可见文字使用英文，中文图题和图注保留在论文正文或绘图计划中。

正式图片采用“`章节号-图序_中文图片名.png`”格式命名。图号和中文图题写入文件名，但不写入图片画布。第三章 3.3 节的预定输出文件如下：

```text
3-1_真实玉米数据组成与匹配流程.png
3-2_SNP质量控制过程中的位点数量变化.png
3-3_LD剪枝后SNP的全基因组分布.png
3-4_二十四个候选表型的纵向变化轨迹.png
3-5_候选表型标准化平均轨迹热图.png
3-6_二十四个候选表型的遗传力分布.png
3-7_候选表型轨迹相似性及层次聚类结果.png
3-8_最终代表性状的个体纵向轨迹.png
```

## 坐标轴命名

坐标轴和色标的名称以 `src/dynags_figures/labels.py` 中的 `AXIS_LABELS` 为准。同一变量跨图使用相同名称：时间为 `Time`，表型值为 `Phenotypic value`，训练样本量为 `Training sample size`，预测误差为 `Prediction error`，预测指标为 `PCC`、时间校正的 `R²` 和 `RMSE`。原始数值尺度、误差方向与统计区间在图注中说明。

单独性状名称、标准化轨迹的 Z 分数、轨迹间相关系数和真实/模拟遗传力符号按各自定义保留，不按字面相近强制替换。新增或修改图件后先运行 `uv run python scripts/check_axis_labels.py` 检查命名，再检查最终图片的布局。

## 环境初始化

```powershell
uv sync
uv run python scripts/check_environment.py
```

## 后续脚本命名

```text
scripts/figure_3_1_data_overview.py
scripts/figure_3_2_qc_attrition.py
scripts/figure_3_3_snp_density.py
scripts/figure_3_4_trait_trajectories.py
scripts/figure_3_5_trajectory_heatmap.py
scripts/figure_3_6_heritability.py
scripts/figure_3_7_similarity_clustering.py
scripts/figure_3_8_selected_trajectories.py
```

完整绘图要求见 `../../07_论文图表/第三章3.3真实玉米数据集绘图计划.md`。

