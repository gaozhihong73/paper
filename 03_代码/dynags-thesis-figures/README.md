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

