# E5 Ablation Tables

Results use 10 independent repeats. R² is time-adjusted R²; RMSE uses the original phenotype scale. E5a is pending. E5b and E5c retain the previously calculated mean ± SD values; their raw E5 runs are no longer present in this workspace.

## Table E5a. Effect of SNP screening

| Model         | ΔPCC ↑ (95% CI)      | ΔR² ↑ (95% CI)       | ΔRMSE ↑ (95% CI)     |
| ------------- | -------------------- | -------------------- | -------------------- |
| Ridge         | 0.160 [0.132, 0.189] | 0.093 [0.070, 0.117] | 0.059 [0.042, 0.076] |
| RR-GBLUP      | 0.136 [0.114, 0.158] | 0.070 [0.056, 0.083] | 0.043 [0.034, 0.051] |
| Random Forest | 0.070 [0.057, 0.082] | 0.032 [0.020, 0.045] | 0.021 [0.013, 0.029] |
| XGBoost       | 0.047 [0.041, 0.053] | 0.031 [0.020, 0.042] | 0.020 [0.013, 0.027] |
| DNNGP-SNP     | 0.199 [0.170, 0.228] | 0.129 [0.108, 0.151] | 0.083 [0.068, 0.098] |
| Cropformer    | 0.321 [0.280, 0.362] | 0.196 [0.170, 0.221] | 0.123 [0.102, 0.143] |
| DynaGS        | 0.216 [0.187, 0.245] | 0.118 [0.096, 0.140] | 0.104 [0.089, 0.119] |

For each model, compare screened SNPs with all 10,000 SNPs on the same repeat. ΔPCC and ΔR² equal Screened minus All; ΔRMSE equals All minus Screened. Positive values indicate a screening benefit. Ridge and RR-GBLUP are traditional methods; Random Forest and XGBoost are machine learning methods; DNNGP-SNP and Cropformer are deep learning methods.

## Table E5b. Architecture ablation

| Mode | PCC ↑ | R² ↑ | RMSE ↓ |
| --- | ---: | ---: | ---: |
| Wide | 0.588 ± 0.069 | 0.369 ± 0.079 | 0.950 ± 0.124 |
| Deep | 0.272 ± 0.230 | 0.080 ± 0.111 | 1.124 ± 0.143 |
| Add | 0.665 ± 0.093 | 0.456 ± 0.133 | 0.864 ± 0.163 |
| All | 0.725 ± 0.037 | 0.534 ± 0.062 | 0.803 ± 0.102 |

All modes use screened SNPs. Add combines Wide and Deep without gating; All includes gating.

## Table E5c. Screening methods

| Method | PCC ↑ | R² ↑ | RMSE ↓ |
| --- | ---: | ---: | ---: |
| GWAS | 0.571 ± 0.106 | 0.362 ± 0.119 | 0.936 ± 0.143 |
| DC-SIS | 0.530 ± 0.110 | 0.297 ± 0.144 | 0.979 ± 0.162 |
| Serial |  0.725 ± 0.037 |  0.534 ± 0.062 | 0.803 ± 0.102 |
| Union | 0.628 ± 0.103 | 0.418 ± 0.137 | 0.881 ± 0.162 |
| Random | -0.018 ± 0.127 | -0.034 ± 0.033 | 1.183 ± 0.141 |

Serial applies GWAS followed by DC-SIS. Union combines the two rankings. Random selects SNPs with a fixed seed per repeat.
