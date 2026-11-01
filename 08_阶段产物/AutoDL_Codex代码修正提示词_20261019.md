# AutoDL Codex：DynaGS 代码修正提示词

适用仓库：`https://github.com/gaozhihong73/Code`  
编写提示词时的最新提交：`1bd06eb90b086ffa1404fb3d73c8a067d461b3b8`（2026-10-19，新增多表型拼接联合筛选）

## 使用方法

1. 在 AutoDL 中进入该仓库，建议在同一个 Codex 会话中依次发送下面的提示词。
2. 先发送第 0 步。只有当前步骤的测试通过、diff 经你检查后，才发送下一步。
3. 每一步完成后，先让 Codex 汇报改动和测试，不要自动 push。确认无误后再单独要求它提交并推送。
4. 仓库会持续更新，所以每一步都要求 Codex先检查当前 HEAD 和工作区，不能机械地覆盖新代码。

---

## 第 0 步：建立当前代码基线，只审计不修改

```text
你正在维护 DynaGS 仓库。先进行只读审计，本步骤不要修改任何文件。

背景：
- 仓库：https://github.com/gaozhihong73/Code
- 我已知的最新提交是 1bd06eb90b086ffa1404fb3d73c8a067d461b3b8，但仓库可能已有更新。
- 研究任务是使用约 400 个玉米个体的 SNP 预测同一个体多个时间点的动态表型。
- TFPP 是主预测模型；GWAS 和 DC-SIS 是监督式 SNP 位点筛选；ConvAE 实际是带表型辅助任务的掩码自编码表征方法。

请完成：
1. 阅读仓库中的 AGENTS.md、README、配置、训练入口、推理入口、数据加载器、trainer、model、utils 和现有测试。
2. 执行 git status、git log -5，并报告当前 HEAD。不要覆盖未提交修改，不要执行 reset、checkout 丢弃修改或 force 操作。
3. 对当前代码逐项核实：
   - TFPP 的普通训练是否先用全部样本表型筛选 SNP、再划分 train/val/test；
   - K 折模式的每一折是否只使用该折训练个体进行筛选和归一化；
   - infer.py 是否重新用全体表型计算筛选，而不是加载训练时保存的索引；
   - 筛选缓存键是否包含训练个体索引的稳定哈希；
   - TFPPConfig、train.py、infer.py 的 screen_multivariate 默认值是否一致；
   - ConvAE 是否在划分前用全体表型拟合归一化；
   - ConvAE 的表型头是否固定输出 14，use_pheno_help 是否真正控制模型和损失；
   - ConvAE 连续特征能否直接进入当前 TFPP 的四类别 Embedding。
4. 找出当前仓库的测试命令和 Python 版本；在不需要真实数据的前提下运行已有快速测试或最小导入检查。
5. 最后只输出：当前 HEAD、工作区状态、问题是否仍存在、涉及文件和函数、可运行的测试命令、建议的修改顺序。不要写代码。
```

通过标准：审计结果明确指出数据划分、筛选、缓存、推理和 ConvAE 的真实调用顺序，并以当前 HEAD 为依据。

---

## 第 1 步：建立统一、可复现的个体划分文件

```text
请在当前 DynaGS 仓库中完成“统一个体划分基础设施”，本步骤只处理划分和索引，不修改 GWAS、DC-SIS、TFPP 或 ConvAE 的核心算法。

开始前：
- 阅读 AGENTS.md 并检查 git status；保留所有已有修改。
- 报告当前 HEAD。如果它晚于 1bd06eb，请以当前代码为准适配。
- 先查看现有 TFPP_DataLoader、ConvAE_DataLoader 和 K 折逻辑，复用现有项目风格。

目标：TFPP、ConvAE 和后续所有对比实验都能读取同一份按个体划分的 train/val/test 索引。同一个体的全部时间点必须始终属于同一集合。

实现要求：
1. 增加独立的划分工具模块，职责包括：
   - 根据样本数或个体 ID、比例和 seed 生成 train_idx、val_idx、test_idx；
   - 保存和加载 `.npz` 划分文件；
   - 保存 seed、比例、样本数、个体 ID 顺序或其稳定哈希、生成时间等元数据；
   - 校验三组索引无交集、无重复、范围合法、完整覆盖预期样本。
2. 如果当前数据没有显式个体 ID，先使用稳定的行索引作为 sample_id，并在元数据中明确记录；不要假装已有生物学 ID。
3. 增加配置或命令行参数，使训练可以指定已有 split 文件，也可以在文件不存在时生成并保存。路径命名要适合不同 seed 和实验复用。
4. 数据加载函数应能接受显式 indices；本步骤保持原有训练行为兼容，不要提前移动筛选逻辑。
5. 写必要的快速测试，至少覆盖：
   - 相同 seed 和输入得到完全相同的划分；
   - 不同集合无交集；
   - 索引合并后覆盖全部样本且无重复；
   - 样本数或 ID 哈希不匹配时拒绝复用旧划分；
   - 保存后加载结果完全一致。
6. 使用小型合成数据运行测试，不依赖真实玉米数据和 GPU。

完成后停止，不要继续修改筛选逻辑。汇报：修改文件、公共 API、测试命令与结果、兼容性影响、建议 commit message。不要自动 push。
```

通过标准：能独立生成、保存、加载、校验同一份个体划分，且快速测试通过。

---

## 第 1A 步：统一筛选使用的基因型编码和筛选配置

第 0 步审计新增发现：当前真实数据加载器在筛选前执行 `+1`，把基因型剂量 `0/1/2` 转为模型类别 `1/2/3`；但 DC-SIS 的闭式计算只统计 `0/1/2`。因此应在移动筛选流程之前，先建立明确的编码边界。

```text
请修正 DynaGS 的 SNP 筛选输入编码，并统一筛选配置默认值。本步骤只处理编码适配、缺失值边界、配置一致性和对应测试；不要移动普通训练/K 折的筛选时机，不要改 TFPP 或 ConvAE 架构。

开始前：
- 检查当前 HEAD、git status，保留已有修改；运行第 1 步的划分测试。
- 阅读 TFPPDataset、TFPP.forward、Screening.py、Sim_DataLoader、train.py、infer.py 和配置，明确各阶段真实编码。

已确认的问题：TFPPDataset 当前将原始 `-1/0/1/2` 执行 `+1`，模型类别变为 `0/1/2/3`；DC-SIS 的闭式实现只统计剂量 `0/1/2`。同一无缺失 SNP 在错误偏移编码下的 dCor 会明显变化。

实现要求：
1. 定义并在代码/文档中明确两种不同表示：
   - screening dosage：供 GWAS/DC-SIS 使用的基因型剂量 `0/1/2`；
   - model categorical code：供 TFPP/ConvAE Embedding 使用的 `0/1/2/3`，其中缺失类别的含义必须明确。
2. 增加一个集中、可测试的编码转换接口，禁止在多个文件中散落 `+1`、`-1` 或 `long()` 来猜测输入语义。
3. screen_snps、gwas_screen、dc_sis_screen 收到的数据必须经过显式校验。不要仅把 DC-SIS 的判断从 `0/1/2` 改成 `1/2/3`，因为这会继续混淆缺失类别和基因型剂量。
4. 对缺失基因型制定显式策略：
   - 先检查仓库是否已有可信且记录在案的处理方式；
   - 若没有，增加只由训练个体拟合的、可配置且可追溯的处理方式，优先使用保持离散剂量的确定性策略；
   - 禁止把 missing 静默当作真实的剂量 0；
   - 本步骤可以实现编码/拟合接口，实际 fit 必须在后续第 2 步的 train_idx 上调用。
5. 保持模拟数据与真实数据使用同一个筛选编码契约，防止两条数据管线含义不同。
6. 统一 `screen_multivariate` 默认值。以 TFPPConfig 为单一默认来源，train.py 和 infer.py 不得复制出相反默认值；保留显式 true/false 覆盖能力。将最终生效值写入日志和保存配置。
7. 让旧筛选缓存失效：在缓存键中加入明确的编码版本和筛选算法版本。此时暂不加入训练索引哈希，训练索引将在第 2 步接入。
8. 添加合成测试：
   - 同一剂量 SNP 经过 model code 编码再显式解码后，GWAS/DC-SIS 得到相同评分和排序；
   - 错误的 `1/2/3` 输入会被校验拒绝，而不是静默产生结果；
   - missing 不会被当成真实剂量 0；
   - 模拟与真实管线给相同 dosage 时筛选结果一致；
   - 配置、训练 CLI、推理 CLI 的 screen_multivariate 默认行为一致。
9. 使用 CPU 合成数据运行测试和最小导入检查，不运行真实数据筛选。

完成后停止。汇报编码契约、缺失值策略、旧缓存如何失效、修改文件、测试结果和建议 commit message。不要自动 push。
```

通过标准：GWAS/DC-SIS 始终接收语义明确的 `0/1/2` 剂量，缺失值不会冒充基因型，配置默认值一致。

---

## 第 2 步：修正普通 TFPP 训练中的监督筛选泄漏

```text
请基于已经完成的统一 split 基础设施，修正 TFPP 普通 train/val/test 模式中的 GWAS/DC-SIS 信息泄漏。本步骤只处理普通划分模式，暂不改 K 折，也不改 ConvAE 和连续特征输入。

开始前检查当前 HEAD、git status，并运行第 1 步和第 1A 步测试。保留 2026-10-19 后加入的多表型联合筛选能力及已建立的筛选编码契约，不要退回旧的逐表型实现。

当前问题：TFPPTrainer 在 dataset 初始化后、数据划分前，对全部样本的 phenotype 执行 screen_snps，测试表型参与了特征选择。

实现目标与要求：
1. 调整普通训练数据流为：
   a. 加载原始 SNP、表型及 sample_id；
   b. 生成或加载固定 split；
   c. 将 `snp[train_idx]` 通过第 1A 步建立的显式接口转换为 screening dosage；若配置为 mode 填补，只允许在 train_idx 上拟合填充值，并只为筛选生成训练剂量矩阵；
   d. 只把训练剂量矩阵和 phenotype[train_idx] 传给 screen_snps；
   e. 用返回的原始列索引切片 train、val、test 的模型类别 SNP；不要因为筛选用过 mode 填补就静默改变 TFPP 的缺失类别输入语义；
   f. 只用 train phenotype 拟合表型 scaler；
   g. 在筛选后得到真实输入维度，再创建 TFPP 模型。
2. 为同一批筛选后的 SNP 明确构造两种输入视图，不能把模型类别码直接当作连续剂量：
   - Deep 视图：`0` 为缺失，`1/2/3` 对应真实剂量 `0/1/2`，只送入类别 Embedding；
   - Wide 视图：先恢复真实剂量；缺失值只使用由 train_idx 拟合并保存的规则处理，再除以 2 得到剂量尺度 `0/0.5/1`（若采用训练均值填补，允许得到区间内连续值）；只送入 Legendre Wide 层。
   当前 `TFPP.forward()` 中对统一类别输入直接执行 `x / 2.0` 的路径必须移除或替换。可以使用显式双输入、共享的基因型预处理器，或注册训练期拟合的填补参数；无论采用哪种设计，类别语义和剂量语义都必须分离，并能随 checkpoint/实验元数据复现。不要改动线性注意力、时间预测头和门控等核心算法。
3. 删除或迁移 TFPPTrainer.__init__ 中“全量筛选后再划分”的逻辑。避免仅在表面上增加 split，但内部仍读取完整 phenotype。
4. 保留并正确传递 screen_k、gwas_k、screen_multivariate。修正目前 TFPPConfig 与 train.py 的 screen_multivariate 默认值不一致：以配置为唯一默认来源，并把这次新增的多元联合模式作为显式、可追溯配置；训练日志和保存的配置必须记录最终生效值。
5. 筛选缓存键必须包含训练个体索引的稳定哈希、筛选模式、表型列表、参数、数据指纹和必要的 fold/split 标识。更换 val/test 表型不能改变缓存键；更换训练个体必须改变缓存键。
6. 每次训练保存：
   - 实际使用的 split 文件副本或引用及哈希；
   - `screened_indices.npy`，索引必须相对于原始 SNP 矩阵；
   - 原始 SNP 维度、筛选后维度、筛选参数、训练个体哈希和筛选缓存键；
   - 筛选剂量编码版本、缺失策略，以及 mode 策略启用时仅由训练个体拟合的填充值/其可验证元数据；
   - 表型 scaler。
7. 不要在筛选前对全体表型做任何依赖标签的统计变换。
8. 添加回归测试：固定 train_idx 后，任意修改 val/test phenotype 或 val/test genotype，screened_indices、训练表型 scaler、筛选缺失填充值和筛选缓存键都必须不变；修改 train phenotype 或 train genotype 时允许结果变化。还要断言 screen_snps 实际收到的只有训练行且编码为合法 dosage 0/1/2。
9. 增加输入语义测试：同一无缺失剂量矩阵应生成正确的 Deep 类别 `1/2/3` 和 Wide 数值 `0/0.5/1`；缺失在 Deep 中保持类别 0，在 Wide 中等于训练期保存的填充值缩放结果；修改 val/test 缺失分布不能改变 Wide 填补参数；模型完整、wide-only、deep-only 三种模式均能前向/反向。
10. 用小型合成数据运行一次 CPU 冒烟测试，覆盖加载、划分、筛选、两类输入视图、模型创建和一个前向/反向步骤。不要运行正式大数据训练。

完成后停止。汇报 diff 摘要、旧泄漏路径如何被切断、测试命令与结果、尚未处理的 K 折问题、建议 commit message。不要自动 push。
```

通过标准：普通训练中任何验证集或测试集表型变化都不会影响筛选结果与训练期 scaler。

---

## 第 3 步：让推理只加载训练产物，不重新用表型筛选

```text
请修正 DynaGS 的 TFPP 推理与模型加载流程，使推理严格复用训练时保存的数据处理产物。本步骤不要修改模型架构，也不要处理 ConvAE。

开始前检查当前 HEAD、工作区以及第 2 步测试。

当前问题：infer.py 的测试集模式会重新构建 dataset，并可能再次用全体 phenotype 调用 screen_snps；外部输入模式也缺少对“原始全维 SNP”和“已筛选 SNP”的明确区分。

实现要求：
1. 推理不得调用任何需要测试表型拟合的筛选或归一化步骤。删除 infer.py 中重新计算 GWAS/DC-SIS 的路径。
2. 从实验目录加载训练时保存的：
   - 模型配置与 checkpoint；
   - split 文件；
   - screened_indices.npy；
   - 表型 scaler；
   - 原始维度、筛选后维度、表型、时间点和输入模式元数据。
   - 基因型编码版本、筛选缺失策略、Wide 通路训练期填充值及相应哈希。
3. 测试集复现模式必须使用训练时保存的 test_idx 和 screened_indices，不能仅凭 seed 重新猜测。
   测试表型如需用于计算指标，只能用训练时保存的 phenotype scaler 执行 transform；不得调用会重新 fit scaler 的通用 DataLoader 路径。反归一化也必须使用同一份保存的 scaler。
4. 外部 SNP 输入增加明确的输入阶段参数，例如 raw 或 screened：
   - raw：检查原始维度，然后应用训练时的 screened_indices；
   - screened：检查筛选后维度，禁止再次筛选；
   - 不允许仅因为维度碰巧相同就静默判断输入阶段。
5. 检查 SNP 编码范围和缺失值约定；异常时给出可操作的错误信息。
   对 raw SNP 应按训练元数据生成 Deep 类别视图与 Wide 剂量视图：Deep 保留缺失类别 0；Wide 使用训练期保存的填充值并按训练时相同规则缩放。推理期间禁止重新计算填充值。
6. 模型加载器必须核对输入维度、输入模式、表型名称和时间点；缺少必要产物时明确失败，不回退到重新拟合。
7. 写测试覆盖：
   - 推理代码不会调用 screen_snps；
   - raw 输入按保存索引得到与训练测试输入相同的列；
   - screened 输入不重复切片；
   - 错误维度、越界索引、缺失产物会失败；
   - 修改测试表型不会改变预测输入。
   - 修改测试集缺失分布不会改变保存的 Wide 填充值；相同 checkpoint、split 和原始输入得到的预测应与训练阶段保存/复现的测试预测在允许误差内一致。
   - 新格式实验若缺少 split、筛选索引、scaler、Wide 填充值或预处理版本，应明确失败。旧 checkpoint 兼容必须通过显式 legacy 选项进入并打印醒目警告，不能静默回退；legacy 结果不得进入论文正式实验汇总。
8. 使用合成 checkpoint/元数据完成 CPU 冒烟测试，不运行正式训练。

完成后停止，汇报新的推理命令示例、兼容性变化、测试结果和建议 commit message。不要自动 push。
```

通过标准：推理完全由训练产物决定，不读取测试表型来筛选特征。

---

## 第 4 步：修正 K 折模式中的折内筛选和模型重建

```text
请修正 TFPP 的 K 折交叉验证，使每一折都独立完成训练期预处理。本步骤不要处理 ConvAE。

开始前阅读当前 K 折代码，不要假定它与旧版本相同。检查当前 HEAD、git status，并先运行普通模式的快速测试。

实现要求：
1. 明确每折的 train/val/test 角色。若现有 KFold 只产生 train/test，请在外层 train 内再确定验证集用于早停，且 test 只用于最终评分。
2. 每一折必须按以下顺序执行：
   - 确定本折 train/val/test 个体索引；
   - 仅使用本折 train genotype 拟合筛选缺失填充值，并生成合法的 screening dosage；仅使用本折 train phenotype 拟合 GWAS/DC-SIS 和表型 scaler；
   - 将筛选索引应用于本折三组 SNP；
   - 为本折三组数据构造与普通训练一致的 Deep 类别视图和 Wide 剂量视图；Wide 只能使用本折训练期填充值；
   - 根据本折筛选后的维度重新创建全新 TFPP、优化器和调度器；
   - 只用 val 早停和选模型；
   - 最后一次性评估 test。
3. 任何模型权重、优化器状态、scaler、筛选索引都不能跨折复用。
4. 缓存键必须包含本折训练个体索引哈希、训练基因型/表型内容指纹和 fold 编号；保存每折 split、screened_indices、scaler、Wide 填充值、配置、checkpoint、逐样本预测和指标。每折实验目录使用与普通训练和新推理兼容的 schema 3 产物及哈希校验，索引必须相对于原始 SNP 矩阵。
5. 汇总结果报告各折指标的均值与标准差，同时保留每折明细。不得只汇报把所有折预测混合后的一个 PCC。
6. 增加泄漏回归测试：固定某折 train_idx，修改该折 val/test phenotype 或 genotype，不得改变筛选索引、scaler、Wide 填充值或训练缓存键；断言每折筛选函数只收到该折训练行。
7. 外层每个样本应恰好作为 test 出现一次；汇总预测必须携带原始 sample_id 和 fold，能够恢复到统一样本顺序。不得用其他折测试标签参与当前折的预处理或模型选择。
8. 用小型合成数据运行至少 2 折 CPU 冒烟测试，验证每折模型对象与预处理产物相互独立；并使用第 3 步的新推理加载其中一个 fold 目录，复现该折保存的测试预测。

完成后停止。报告当前 K 折定义、修改文件、产物目录、测试结果和建议 commit message。不要自动 push。
```

通过标准：每折筛选和归一化只依赖该折训练个体，并且每折模型从头初始化。

---

## 第 5 步：修正 ConvAE 的数据划分与表型归一化

```text
请修正 ConvAE 数据管线中的表型泄漏，并让它复用 TFPP 的统一个体 split。本步骤先不增加 TFPP 连续输入头，也不导出最终潜在特征文件。

背景：ConvAE 是代码名称；当前实现本质上是带表型辅助任务的 masked autoencoder，不要为了符合名称而擅自加入卷积层或大范围改名。

实现要求：
1. ConvAEDataset 只负责加载原始 SNP、原始 phenotype、sample_id，以及确定分块/padding 信息；Dataset 构造函数中不得使用全体 phenotype 计算均值和标准差。
2. ConvAE 读取与 TFPP 相同的外层 train/val/test split。仅用 train phenotype 拟合表型 scaler，再变换 train、val、test。
   - 普通模式直接读取 `IndividualSplit`；
   - 需要与 K 折比较时，接受 TFPP 已保存的某个 fold manifest 中的显式 train/val/test 索引，禁止重新随机划分；
   - sample_id 顺序与 split/fold 哈希必须写入 ConvAE 实验元数据。
3. test phenotype 不得用于训练、早停、归一化拟合或超参数选择。val 只用于早停和模型选择。
4. 保存 ConvAE 使用的 split 引用/哈希和 phenotype scaler。输出目录应带 experiment 和 split/fold 标识，避免不同划分覆盖。
5. 如果表型存在缺失值，先检查当前代码的处理约定；不要用 test 统计量填补。缺少明确约定时应报错或使用只由 train 拟合的策略，并在元数据记录。
6. 明确区分“真实基因型缺失”和“为补齐最后一个 SNP chunk 添加的结构性 padding”。Dataset 应提供可复用的有效位置或 padding mask 及原始有效长度；不能只因为二者都使用类别 0 就丢失语义。此步骤先建立数据接口和测试，模型如何在注意力、池化和重建损失中消费该 mask 放在第 6 步完成。
7. 修改 dataloader 和 trainer 接口时保持旧命令尽可能兼容；无法兼容的地方在 README 和错误信息中明确说明。
8. 添加测试：
   - train/val/test 无交集；
   - scaler 参数只由 train phenotype 决定；
   - 修改 val/test phenotype 不改变 scaler；
   - 相同 split 在 TFPP 和 ConvAE 中得到完全相同的 sample_id 顺序。
   - 结构性 padding mask 与原始 SNP 有效长度一致，真实缺失位点不会被误标为 padding；不同 chunk_size/overlap 下仍正确。
9. 保存的数据管线元数据沿用当前实验产物的 schema 3 哈希原则，至少校验 split/fold、样本顺序、表型、时间点、基因型编码版本、原始有效长度和 chunk 参数。
10. 用小型合成 SNP 和 phenotype 完成 CPU 数据管线测试，不运行完整 ConvAE 训练。

完成后停止。汇报原泄漏位置、修改后的数据流、测试结果和建议 commit message。不要自动 push。
```

通过标准：ConvAE 的所有表型统计量只由训练个体拟合，并能复用同一份 split。

---

## 第 6 步：修正 ConvAE 输出维度、辅助任务开关和模型选择

```text
请修正 ConvAE 模型与训练器的一致性问题，并让模型正确消费第 5 步新增的结构性 padding mask。本步骤保持类名和现有 checkpoint 兼容策略清晰，不增加 TFPP 连续输入头。

开始前检查当前 HEAD、git status，并运行第 5 步数据管线测试。

实现要求：
1. 移除 phenotype_head 最后一层固定输出 14 的假设。模型构造时显式接收 n_phenotypes 和 n_time_points，输出恢复为 `(batch, n_phenotypes, n_time_points)`；单表型 14 时间点仍应正常工作。
2. phenotype 数据列数、配置 time_coordinates 和模型 n_time_points 必须一致，否则尽早报错。
3. 让 use_pheno_help 真正控制：
   - 是否构建/调用表型辅助头；
   - pheno loss 是否参与总损失；
   - 对应参数是否进入优化器；
   - 日志和保存配置是否记录开关。
4. 明确定义并实现模型选择指标：开启表型辅助时固定使用验证总损失，关闭时固定使用验证重建损失；训练与验证中的各损失定义必须一致。验证重建指标不能恒为零，验证掩码必须在给定 seed、split 和 sample_id 后固定可复现。
5. recon_weight 和 pheno_weight 的合法性应校验；关闭表型辅助时不能仍保留非零 pheno loss。
6. 为旧 checkpoint 提供明确的加载错误或兼容迁移说明，不要静默加载形状不匹配的权重。
7. 将第 5 步的数据 mask 明确接入模型和损失：
   - 结构性 padding 不得参与注意力、全局池化、随机掩码采样或重建损失；
   - 最后一个不完整 chunk 的有效 SNP 必须正常参与计算；
   - 真实基因型缺失不能被当作结构性 padding。它可以保留为独立缺失类别参与编码，但由于没有真实剂量标签，不得作为重建监督目标；
   - 重建损失只在“被选中重建、非结构性 padding、原始基因型非缺失”的位置计算，并正确处理分母；如果一个 batch 没有可用重建目标，应采用明确、可测试的处理方式，不能产生 NaN；
   - 验证阶段使用固定且可复现的掩码，不能因每轮随机掩码变化而让早停指标不可比较。训练阶段可以继续使用受 seed 控制的随机掩码。
8. 固定模型选择口径，避免后续根据结果更换：`use_pheno_help=true` 时使用与训练目标同定义的 `val_total_loss = recon_weight * val_recon_loss + pheno_weight * val_pheno_loss`；`use_pheno_help=false` 时使用 `val_recon_loss`。将该口径和两个分项写入日志、history、checkpoint metadata 和配置。
9. 测试至少覆盖：单/多表型输出形状、有/无表型辅助、一次前向反向、验证指标非零且可复现、配置不一致时报错、checkpoint 保存加载；还要覆盖 padding 改变不影响有效位点输出/池化、padding 不进入重建损失、真实缺失不被误判为 padding、最后一个不完整 chunk、全 padding 或无可用重建目标的安全处理。
10. 只使用小型合成数据进行 CPU 测试。完成一个包含非整除 SNP 数、结构性 padding 和真实缺失的单轮训练冒烟测试；不得运行正式数据训练。

完成后停止。汇报模型选择标准、接口变化、测试结果和建议 commit message。不要自动 push。
```

通过标准：ConvAE 支持任意配置内的表型数和时间点数，辅助任务开关真实有效，早停指标有定义且可复现。

---

## 第 7 步：安全导出 ConvAE 的 train/val/test 潜在特征

```text
请实现 ConvAE 潜在特征的可追溯、确定性导出。本步骤只导出表征，不修改 TFPP 模型，也不运行正式数据训练。

实现要求：
1. 只能通过第 6 步新增的严格 `load_convae_model()` 加载当前 split/fold 上由 train 训练、由 val 选出的最佳 checkpoint。checkpoint 中的 split/fold 哈希、模型配置和预处理 schema 必须与当前导出任务一致；不一致时明确失败。
2. 增加明确的 `encode_features()` 或等价接口。导出时使用完整、未随机遮蔽的基因型输入，即重建 mask ratio 为 0；仍必须传入并消费结构性 padding mask。不得复用训练或验证阶段的随机 reconstruction mask。
3. 编码 train、val、test 时只能输入 SNP、padding/有效位置 mask 和 sample_id。不得把 phenotype、表型预测值或表型 scaler 传入编码函数或参与特征生成；即使 `use_pheno_help=true`，导出也只调用编码器路径，不调用表型辅助头。
4. 固定导出为二维特征矩阵 `(N, D)`，不得在看到实验结果后改变表示方式。如果编码器原生输出 `(N, num_chunks, d_ext)`，按固定 chunk 顺序展平为 `(N, num_chunks * d_ext)`，不新增可训练池化层；记录 latent_layout、展平顺序和最终 D。最后一个含有效 SNP 的不完整 chunk 必须保留，纯结构性 padding 不得影响其表示。
5. 分别保存 `features/train.npy`、`features/val.npy`、`features/test.npy` 以及对应的 `sample_ids`。最终行顺序必须严格等于 split/fold manifest 中的顺序；即使 DataLoader 改变 batch size 或内部顺序，也要按 sample_id 恢复顺序，并拒绝重复、缺失或额外 sample_id。
6. 每个 split/fold 和 checkpoint 使用不会互相覆盖的导出目录。保存 metadata.json，至少包括：
   - 当前 git commit；
   - split 文件路径和哈希；
   - train/val/test sample_id 顺序或哈希；
   - 原始 SNP 维度、padding、chunk_size、overlap、num_chunks、d_ext 和最终潜在维度；
   - latent_layout、展平规则、数组 shape 和 dtype；
   - 表型与时间点；
   - ConvAE checkpoint 路径、文件哈希、模型配置和模型选择指标；
   - use_pheno_help、随机种子、数据文件指纹和导出 schema 版本；
   - 每个特征及 sample_ids 文件的内容哈希。
7. 导出函数必须使用 `model.eval()`、`torch.inference_mode()` 或 `torch.no_grad()`，关闭训练期随机遮蔽，并保证相同 checkpoint 和输入得到相同结果。完成写入后重新加载文件，校验 shape、有限值、sample_id 顺序和内容哈希。
8. 增加测试：
   - 修改 train/val/test phenotype 均不改变对应导出特征；
   - 重复导出以及改变 batch size 后结果在允许浮点误差内一致；
   - 改变结构性 padding 中的填充值但保持 mask 不变，不影响有效样本特征；
   - sample_id 与特征行一一对应，乱序 loader 能恢复 manifest 顺序，重复/缺失 ID 明确失败；
   - checkpoint、split 哈希或配置不匹配时拒绝导出；
   - 不同 split/fold/checkpoint 不会覆盖输出。
9. 使用小型合成数据训练一个最小 checkpoint 并完成 train/val/test 导出冒烟测试，给出 CLI 命令；不得运行正式大数据训练或压缩。

完成后停止。汇报 encode_features 接口、特征 shape、输出目录、metadata 示例、确定性与表型隔离测试、完整测试结果和建议 commit message。不要自动 push。
```

通过标准：导出的潜在特征只由基因型和训练好的编码器决定，行顺序及训练来源均可追溯。

---

## 第 8 步：先建立 ConvAE 表征的独立下游基线

```text
请为 ConvAE 潜在特征增加一个简单、固定、可复现的 Ridge 下游预测基线，用于先判断表征是否包含动态表型预测信息。本步骤不要改 TFPP，不运行正式数据，不比较或尝试 MLP，也不要为了提高结果扩大调参范围。

要求：
1. 模型固定为多输出 `Ridge`：输入为 ConvAE 导出的二维连续特征 `(N,D)`，将目标暂时展平为 `(N, P*T)` 训练，预测后恢复为 `(N,P,T)`。固定 `fit_intercept=true`、`solver=lsqr`、`tol=1e-6`、`max_iter=10000`，不得根据 test 结果更换模型或 solver。
2. alpha 候选集合预先固定为 `[1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000, 10000]`。每个 alpha 只在 train 上拟合，以归一化尺度下的整体 val MSE 选择最小者；数值并列时固定选择较大的 alpha。test 标签不得参与 alpha、scaler 或任何模型选择。
3. 选定 alpha 后保持该 train-only 模型，不将 val 合并回训练集，不重新拟合；test 只预测和评估一次。记录 test 评估次数或等价状态，防止搜索过程中读取 test 指标。
4. 严格读取并校验第 7 步 `metadata.json` 和所有数组哈希。train/val/test 特征和 phenotype 必须按保存的 sample_ids 对齐；重复、缺失、额外或错序 ID 必须明确失败。
5. 连续特征 scaler 只在 train features 上拟合并保存，应用到 val/test。表型 scaler 也只能在 train phenotype 上拟合并保存，应用到 val/test；反归一化后再计算原始量纲指标。不得复用或重新拟合 test 统计量。
6. 使用与 TFPP 相同的指标函数和聚合口径，输出 PCC、R²、MSE、RMSE、MAE：至少保存整体、逐表型、逐时间点以及逐表型逐时间点结果。R² 不得截断为 0；PCC 遇常数向量时保留 NaN 并记录有效计数，不能静默替换为 0。
7. 保存逐样本逐表型逐时间点的真实值和预测值，必须包含原始 sample_id、split/fold、phenotype 和 time point。预测文件顺序与 test manifest 一致。
8. 输出目录必须由 ConvAE export metadata 哈希和基线配置哈希唯一确定，并禁止覆盖。至少保存：`config.json`、alpha 搜索记录、feature scaler、phenotype scaler、Ridge 模型、`metrics.json`、预测的 NPZ/CSV、运行日志和环境摘要。
9. 记录随机种子、代码 commit/dirty 状态、split/fold 哈希、ConvAE checkpoint 与 export metadata 哈希、特征 shape、训练时间、推理时间和峰值内存。加载已保存模型后必须能复现 test 预测。
10. 该结果只能描述为“ConvAE 表征的独立线性探针/下游基线”，不能写成与 TFPP 完全同构的模型比较，也不能据此声称 ConvAE 优于监督筛选。
11. 合成测试至少覆盖：
   - feature scaler、phenotype scaler、Ridge 和 alpha 选择均不接触 test；
   - 修改 test phenotype 不改变 alpha、scaler、模型和预测，只允许 test 指标变化；
   - 修改 val phenotype 可以影响 alpha，但不能改变 train-only scaler；
   - 样本 ID 对齐和哈希校验；
   - 固定输入重复运行及保存加载后预测一致；
   - test 只评估一次；
   - 指标 shape、反归一化、负 R² 和常数向量 PCC 的处理正确；
   - 不同 split/fold/export 不覆盖。
12. 使用第 7 步的小型合成导出产物完成 CPU CLI 冒烟测试，不运行真实数据基线。

完成后停止。汇报 alpha 选择结果、数据与 scaler 拟合范围、指标层级、输出目录、测试命令、完整测试结果和建议 commit message。不要自动 push。
```

通过标准：在不改 TFPP 的情况下，可以无泄漏地评价 ConvAE 表征。

---

## 第 9 步：可选——为 TFPP 增加连续特征输入模式

只有第 8 步表明 ConvAE 表征值得继续使用时，再发送本提示词。

```text
请为 TFPP 增加显式的连续特征输入模式，同时保持现有离散 SNP 模式行为不变。开始前先确认 ConvAE 独立下游基线已经通过测试。

实现要求：
1. 在配置、训练、模型加载和推理中增加 `input_mode`，至少支持：
   - categorical_snp：当前 0/1/2/缺失类别输入；
   - continuous_feature：ConvAE 导出的连续潜在特征。
2. 把输入逻辑封装为两个清晰模块：
   - categorical_snp 保留四类别 Embedding、整数范围检查和当前局部卷积；
   - continuous_feature 不执行 `+1`、`long()` 或类别 Embedding，先用只在 train features 上拟合的 scaler，再用可学习投影得到 `(B, L, d_model)`。
3. Wide 通路：
   - categorical_snp 保持经过核实的基因型数值变换；
   - continuous_feature 直接使用训练期标准化后的连续值，不能套用离散 SNP 专用的 `/2.0`。
4. 后续线性注意力、时间预测头、门控和输出形状尽量共享。
5. checkpoint/model_config 必须保存 input_mode、原始输入维度、特征 scaler、ConvAE metadata 哈希和必要的输入头参数。加载时严格校验。
6. 推理必须根据保存的 input_mode 选择路径，禁止根据 dtype 或数值范围静默猜测。
7. 测试覆盖两种输入模式的前向、反向、保存加载和推理；确认 categorical_snp 在固定 seed 下与修改前输出一致或在可解释范围内一致；连续模式不得发生整数截断或 Embedding 越界。
8. 用小型合成数据完成 CPU 冒烟训练，不运行正式实验。

完成后停止。报告架构变化、向后兼容结果、测试结果和建议 commit message。不要自动 push。
```

通过标准：两种输入路径显式分离，离散模式无回归，连续模式没有隐式整数化。

---

## 第 10 步：实验运行与结果追溯基础设施

```text
请为 DynaGS 增加统一、配置驱动的实验运行和结果记录基础设施。本步骤只建立运行、校验和记录能力，不启动耗时正式实验，也不实现第 9 步的 continuous_feature TFPP。

需要支持的路线：
1. 预先给定的固定 SNP 索引或由 seed 确定的 train-label-free 随机 k SNP + TFPP；
2. GWAS + TFPP；
3. DC-SIS + TFPP；
4. GWAS→DC-SIS + TFPP；
5. ConvAE 表征 + 固定下游基线；
6. 仅当第 9 步将来完成时，配置 schema 才允许 ConvAE 表征 + continuous_feature TFPP；当前必须明确拒绝该路线，不能静默回退到离散输入。

要求：
1. 所有路线必须使用同一份 split 或同一组外层 folds；禁止每种方法重新随机划分。普通 split 与 K 折结果不得混在同一个统计汇总中。
2. 建立版本化的 experiment spec/schema，显式记录：数据路径和内容指纹、sample ID 来源及顺序哈希、表型、时间点、seed、split/folds、路线、筛选方法及参数、实际输入维度、模型配置、损失、早停标准、指标口径和代码 commit/dirty 状态。正式运行默认拒绝 dirty 工作区，可由仅用于开发 smoke 的显式开关放宽并记录。
3. 实验轴至少支持但不自动展开全部组合：
   - screening route：fixed_indices、random_k、GWAS、DC-SIS、GWAS→DC-SIS、ConvAE→Ridge；
   - TFPP ablation：full、wide_only、deep_only，以及代码确实已实现且有测试的门控消融；未实现的消融必须校验失败，不能伪装成 full；
   - train size：支持固定个体数或 fraction。只从原训练集取子集，val/test 保持完全不变；同一 seed/fold 下按稳定 sample_id 顺序生成嵌套子集，使较小样本集合严格包含于较大集合，并保存 subset manifest/hash。每个子集内部重新拟合筛选、填补和 scaler；
   - trait set、seed 和 fold。
4. 随机 SNP 基线只能依赖训练基因型维度和预先固定的 seed，不能读取任何 phenotype；保存实际索引。fixed_indices 必须校验原始列范围、唯一性、顺序和文件哈希。
5. test 只能在训练、超参数选择和最佳 checkpoint 固定后评估一次。runner 不得在搜索循环或失败重试中读取 test 指标。实验失败时保留错误日志、状态和完整配置，不能用空值或旧结果替代。
6. 每个实验使用由 canonical config hash 唯一确定的目录；相同 hash 默认拒绝覆盖。提供明确的 resume 语义：只允许从完全相同的 config、split、数据指纹和代码版本恢复，否则失败。
7. 每个成功运行至少输出：状态文件、resolved config、manifest、训练历史、筛选/特征产物引用、逐样本逐表型逐时间点预测、整体/逐表型/逐时间点/逐表型逐时间点指标、每折指标及均值和标准差、训练/验证/推理/筛选时间、峰值 GPU 显存与 CPU RSS、模型参数量、实际输入维度及产物哈希。
8. 实际选中的 SNP 数必须从 `screened_indices` 产物读取，不能只记录请求的 k。多表型逐性状并集和联合筛选必须使用不同 method id，并记录每个性状贡献、去重前后数量和最终有序索引哈希。
9. 汇总器只能读取 schema 校验通过且状态为 success 的实验；缺少 split、预测、sample_id、commit、配置哈希、索引/特征哈希或指标的运行不得纳入汇总。聚合时保留 fold、seed、train size、trait set、screening route 和 ablation，不得把不同实验单位混合求均值。
10. 提供 `dry-run`：解析配置、展开计划、校验依赖、显示将运行的实验及输出路径，但不加载大数组、不训练、不写伪结果。另提供小型合成 smoke，实际走通当前已实现的路线；ConvAE→Ridge 可复用第 7/8 步合成产物。
11. 合成测试至少覆盖：所有路线共享 split、嵌套训练子集、random_k 不依赖 phenotype、各子集独立拟合预处理、配置 hash 稳定、不同轴不覆盖、错误 resume 被拒绝、不完整/失败结果不进入汇总、test 标签变化不影响训练期 manifest 与产物、未实现 continuous_feature 和消融被明确拒绝。

完成后停止。给出 experiment spec 示例、dry-run 与 smoke 命令、输出目录示例、结果 schema、已支持和明确拒绝的路线、完整测试结果及建议 commit message。不要自动 push。
```

通过标准：所有后续结果都能追溯到代码、数据、划分、筛选索引、配置和逐样本预测。

---

## 第 11 步：最终防泄漏审计与文档同步

```text
请对当前 DynaGS 仓库做一次实验前最终审计，并修复审计中确认的问题。不要运行真实数据，不要为了清理代码而改动已经验证正确的核心算法或模型结构。

审计重点：
1. 开始前记录当前 HEAD、最近提交、工作区状态和环境版本。确认第 8、10 步已分别提交；如果仍混有未提交的前序步骤，停止并报告，不要把多个步骤合并为一次审计提交。
2. 搜索所有使用 phenotype、scaler、screen_snps、GWAS、DC-SIS、ConvAE fit/export、Ridge、train_test_split、KFold、test metrics 和 experiment summarize 的位置，按入口画出普通 TFPP、K 折 TFPP、ConvAE、ConvAE→Ridge、推理和统一 runner 的真实调用链及 train/val/test 数据流。
3. 核实统一 runner 调用的是已测试的生产训练、筛选、导出和推理实现，而不是一套只在 smoke 中成立的重复或占位实现。六条路线至少通过合成端到端测试到达各自真实核心函数；记录调用证据。GWAS→DC-SIS 必须实际串联，ConvAE→Ridge 必须严格消费匹配的 export。
4. 验证普通训练、K 折、不同 train size、ConvAE、Ridge、推理和汇总均满足：修改 test phenotype 只能改变最终真实标签及相应 test 指标，不能改变 split/subset、筛选、填补、scaler、权重、模型选择、alpha、编码器、特征或预测。修改 test genotype 可以改变该样本预测，但不能改变任何训练期产物。增加或保留对应回归测试。
5. 验证 train-size 实验只缩减 train，val/test 固定；同 seed/fold 的子集严格嵌套；每个子集重新拟合监督筛选、缺失填补和 scaler。不同路线在同一实验单元必须共享完全相同的 split、subset 和 sample_id 顺序。
6. 审计 test 一次性使用约束：训练、早停、alpha 选择、失败重试和 resume 均不能读取 test 指标；选定模型后只能产生一次正式 test 评估。确认失败状态、test 已预约状态和 resume 行为不会允许反复观察 test 后重试。
7. 检查训练、推理、导出和 runner 的配置默认值一致，尤其是 screen_multivariate、screen_k、gwas_k、缺失策略、模型类别编码、表型、时间点、seed、ablation 和 input mode。尚未实现的 continuous_feature TFPP 与未知消融必须明确失败。
8. 检查所有路径和哈希不会在不同数据、seed、fold、train size、trait set、筛选方法、ablation、ConvAE checkpoint 或配置间覆盖。校验 sample_id 顺序、原始列索引、特征、scaler、checkpoint 和预测的保存/加载闭环；损坏或不匹配产物必须失败。
9. 核实所有路线使用同一套指标函数、反归一化和聚合定义。保留负 R²；常数向量 PCC 为 NaN 并记录有效计数；普通 split、fold、seed、trait set、train size、route 和 ablation 不得错误混合求均值。标准差 ddof 必须在 schema 和文档中明确。
10. 审计 fixed_indices、random_k、多表型逐性状并集、联合筛选的 method id 和元数据；random_k 不得读取 phenotype，最终实际 SNP 数和索引必须从产物核实。确认 DC-SIS 接收剂量 0/1/2，而 TFPP Deep 接收模型类别 0/1/2/3，Wide 接收训练集填补后的剂量并缩放到 [0,1]。
11. 对推理做最终回放检查：新 checkpoint 只能使用训练保存的 split、筛选索引、位点顺序、填补规则和 scaler，不得重新监督筛选或重新拟合；旧 checkpoint 必须继续显式标记为不具备论文结果资格。
12. 运行完整快速测试、核心模块静态导入、所有 CLI `--help`、dry-run 零写入检查、六路线合成 smoke、保存加载/推理复现以及所有泄漏回归测试。不要运行真实 SNP 或表型数据。
13. 更新 README：
   - 正确区分 GWAS/DC-SIS 位点筛选与 ConvAE 连续表征；
   - 说明训练折内拟合规则；
   - 给出普通训练、K 折、ConvAE 训练/导出、Ridge、推理、experiment dry-run/smoke/run/summarize 的经过验证命令；
   - 说明旧 checkpoint/旧实验结果是否兼容以及为何不能直接写入论文；
   - 明确当前 sample_id 为稳定行索引，没有生物学 ID，数据文件及顺序改变会被指纹校验拒绝。
14. 新增 `docs/EXPERIMENT_READINESS.md`，列出审计 HEAD、环境、调用链摘要、测试命令和结果、每条路线的 ready/not ready 状态、尚存限制、正式实验前必须准备的真实路径/哈希/时间列/性状，以及第 9 步是否仍保持未启用。不得写入未经运行的真实指标。

完成后停止。不要运行正式实验，不要自动提交或 push。汇报发现及修复的问题、生产调用链证据、修改文件、完整测试结果、各路线 readiness、剩余风险和建议 commit message。
```

通过标准：合成泄漏测试全部通过，README 与真实实现一致，正式实验所需产物均能保存和复用。

---

## 第 12 步：真实数据只读体检与数据清单冻结

```text
请对即将用于 DynaGS 论文实验的真实基因型和表型数据进行只读 preflight，并生成可复用的数据 manifest。本步骤不得训练模型、不得运行监督筛选、不得生成正式 test 指标，也不得修改原始数据。

本项目本次指定的唯一正式原始输入为：
- SNP：`dataset/maize/snp/snp_2860770.npy`
- phenotype 目录：`dataset/maize/pheno/`，其中应包含 24 个性状 CSV。

`snp_50000.npy`、`snp_mi_*.npy`、`sim_*.npy`、`snp_3/`、`pheno_3/`、`cleaned_pheno/` 和 simulated_data 均不是本次正式原始输入，不得自动替代。`all_phenotype_h2.csv` 与 `gene_snp_mapping.csv` 只能作为辅助资料登记，除非用户之后明确说明，否则不进入训练输入。文件名中的 2860770 不能代替真实 shape 检查；此前材料中的 267 万也不得继续直接引用。

用户已于 2026-10-22 明确确认：`snp_2860770.npy` 的第 i 行与 `dataset/maize/pheno/` 下 24 个 CSV 的第 i 行对应同一个体，且 24 个 CSV 的个体顺序完全一致；当前不存在个体 ID 或顺序映射文件。manifest 中必须将证据类型记录为 `user_source_order_attestation`，不能写成已通过生物学 ID 匹配验证。该确认只适用于上述文件的当前内容哈希；任一文件内容或顺序变化后必须重新确认。

用户同时确认时间坐标 `X10...X101` 表示播种后天数，manifest 和文档统一记录为 `days_after_sowing`（DAS），数值坐标为 `[10,12,14,15,17,19,24,26,37,44,51,65,80,101]`。

根据用户提供的《数据预处理.md》，`snp_2860770.npy` 的上游步骤一应登记为：从 `merge.nonCTG.gt.400.vcf.gz` 出发，将位点 ID 设为染色体_位置，排除重复 ID/位置，仅保留 A/C/G/T SNP 和二等位位点，过滤 SNP 缺失率 >5%、MAF <5%、HWE p<1e-6，随后按 100 kb 窗口、步长 1、r²=0.2 做 LD pruning，并导出加性 0/1/2 矩阵。必须区分证据级别：0/1/2 编码、MAF 和缺失率可由当前 NPY 扫描核实；重复排除、SNP 类型、二等位、HWE 和 LD 参数只能标为 `documented_upstream_provenance`，不能声称从 NPY 直接验证。旧文档步骤二的 PCA/RINT/Mean/Slope/AUC/Min-P GWAS TopN 筛选及其后续导出已经废弃，`snp_50000.npy` 等旧预筛选结果不得用于当前正式实验或论文指标。

开始前：
1. 确认当前 HEAD 是第 11 步审计提交，工作区干净，完整测试已通过。
2. 记录 Python、PyTorch、CUDA、GPU、CPU、RAM、磁盘可用空间和当前代码 commit。

实现与检查要求：
1. 优先复用现有校验与完整性工具；如缺少入口，增加一个只读 CLI，例如 `inspect_real_data.py`。大数组必须使用 `.npy` header、mmap 和分块统计，禁止为了体检一次性复制 267 万 SNP 到内存。
2. 对基因型文件记录并校验：绝对路径或可解析路径、完整 SHA-256、文件大小、shape、dtype、C/F order、个体数、SNP 数、允许的原始剂量集合、NaN/-1 缺失数量与比例、逐列缺失率/MAF/方差的分布摘要、常量列数量。这里只做描述和完整性检查，不得用全体个体统计结果生成最终筛选索引；正式的缺失填补、MAF/QC 或监督筛选仍必须在每个训练 split/fold 内拟合。
   - 不要根据文件名假定矩阵方向。必须将两个维度与 phenotype 行数核对；若疑似转置，只报告并停止，不得自动转置或重写文件。
3. 枚举全部 phenotype 文件和可用性状，不依赖人工猜测文件名。对每个性状记录：文件哈希、shape、dtype、时间列原始名称、解析后的时间坐标及顺序、个体数、缺失/非有限值、每个时间点的描述统计。不得计算 GWAS、DC-SIS、模型精度或根据 test 表现选择性状。
4. 检查 SNP 与所有 phenotype 的个体对应关系：
   - 搜索真实生物学 ID 列或外部映射文件；存在时验证唯一性、集合一致性和顺序，并优先使用生物学 ID；
   - 如果确实只有行号，生成冻结的 `sample_ids = 0..N-1`、顺序哈希和每个源文件哈希，并在报告中标记“只能由源数据顺序及用户确认保证对应”；
   - 缺少 ID 时不能声称已从数据本身证明行对齐。本项目已有上述用户确认，因此当前文件哈希完全匹配时可将状态记为 `confirmed_by_user_source_order`；不得记为 `verified_by_biological_id`。哈希变化后状态自动失效并重新成为阻塞项。
5. 时间列必须自然排序并显式保存，例如避免字符串顺序把 T10 放在 T2 前。重复、无法解析、缺失或不同性状时间网格不一致时停止并报告，不得静默重排或截断。
6. 性状选择只依据论文研究目标、生物学意义、测量质量和预先声明的数据完整性门槛。输出 24 个性状的候选表，但不要由脚本擅自决定最终主要性状，也不要依据任何预测结果排序。为每个性状给出可用时间点数、缺失率和需要用户/导师确认的说明。
7. 创建版本化 `real_data_manifest.json`，至少包括：schema、代码 commit、创建时间、数据路径、完整内容哈希、shape/dtype、基因型编码、sample ID 来源与顺序哈希、性状与时间列、缺失摘要以及行对齐确认状态。manifest 使用 canonical JSON 计算自身哈希；原始绝对路径与可能敏感信息不得写入公开示例。
   - 另外保存 `time_unit=days_after_sowing`、结构化 upstream QC 参数、每项 QC 的 evidence level，以及 `deprecated_preselection` 列表，防止旧 50000 位点文件被误纳入当前流程。
   - 用户已确认 `dataset/maize/gene_snp_mapping.csv` 来自质控前数据，与当前 NPY 的 2,860,770 列不对应。将其登记为 `pre_qc_incompatible_mapping`，禁止用于按列映射、候选基因解释、训练或筛选，也不得通过截断、重新排序或模糊匹配补救。
   - 只读寻找生成当前矩阵时与 `data_ld_pruned` 同源且保持最终列顺序的 `.pvar`，或 `data_plink_snp.raw` 的完整表头及等价有序位点列表。只有在位点数量、唯一性、顺序来源和文件哈希均能严格证明与 NPY 列一一对应时，才能冻结为 post-QC SNP mapping；否则将“筛选位点的染色体位置和候选基因解释”列为未解决限制。该限制不阻止以稳定原始列索引进行预测实验。
8. 实际 manifest 和扫描报告保存到被 Git 忽略的 `outputs/preflight/<manifest_hash>/`。仓库只提交 CLI、schema、合成测试、脱敏示例和 README；不得提交真实数据、真实 phenotype 数值、绝对服务器路径或可反推个体的信息。
9. 为后续实验生成但不执行：
   - 一个普通 split/fold 创建命令模板；
   - 一个使用真实 manifest 的 experiment dry-run 配置模板；
   - 明确列出仍需填写的主要性状、时间列、K 折数、seed、候选 SNP 数和输出根目录。
   - 给出本次真实扫描命令，参数必须显式使用上述 SNP 文件和 phenotype 目录，例如：`python -B inspect_real_data.py --snp_file dataset/maize/snp/snp_2860770.npy --pheno_dir dataset/maize/pheno --output_root outputs/preflight`。
10. 用小型合成数据测试：流式哈希与分块统计、剂量/缺失校验、自然时间排序、ID 对齐/错序/重复拒绝、manifest canonical hash、真实路径脱敏、禁止覆盖以及大数组 mmap 路径。测试不得依赖真实数据。
11. 对真实数据只执行只读扫描。若发现形状、编码、缺失、时间列或行对齐问题，立即保留报告并停止，不得自动修复、改写或筛除原始数据。

完成后停止。不要提交或 push，不要开始模型训练。汇报：数据文件及哈希摘要、基因型 shape/编码/缺失、性状与时间点清单、样本 ID 和行对齐证据、阻塞项、manifest 路径及哈希、资源评估、合成测试和真实只读扫描命令、建议 commit message。
```

通过标准：真实数据的内容、顺序、编码、性状和时间列均有冻结记录；任何不能证明的行对齐问题被明确阻止进入正式实验。

---

## 第 13 步：真实数据 pre-test 运行冒烟

```text
请使用已经冻结的真实数据做一次仅用于工程验证的 pre-test smoke，目标是确认真实数据读取、训练折内缺失处理、筛选、GPU 前向反向、checkpoint 和产物记录可以运行。本步骤不得读取 test phenotype、不得计算 test 预测或指标，所有输出必须标记 `paper_eligible=false` 和 `purpose=engineering_pretest_smoke`。

前提与输入：
1. 当前 HEAD 必须是第 12 步提交，工作区干净，完整测试通过。
2. SNP 固定为 `dataset/maize/snp/snp_2860770.npy`，必须验证 manifest SHA-256 与冻结值一致；phenotype 使用 `dataset/maize/pheno/RGBVI.csv`，时间坐标为播种后天数 `[10,12,14,15,17,19,24,26,37,44,51,65,80,101]`。
3. 当前允许使用稳定行索引 `0..399` 和用户的源顺序确认。post-QC SNP ID 映射暂缓，不得使用质控前 `gene_snp_mapping.csv`，训练产物继续保存原始列索引。
4. 缺失策略显式使用 `mode`；所有众数、筛选、基因型/表型 scaler 只能在当前训练个体上拟合。Deep 仍保留缺失类别，Wide 使用训练集众数填补后的剂量并缩放到 `[0,1]`。

实现要求：
1. 先检查现有 runner 是否支持不接触 test 的 `pretest-smoke` 或等价阶段。如果没有，增加一个最小且明确的模式：只创建 train/val loader，完成训练和验证后停止；不得创建 test loader、读取 test phenotype、调用 test metric、写 test 预约状态或生成 test_predictions。不要复制一套新的训练算法，必须复用生产训练、筛选和 checkpoint 路径。
2. 创建一次固定 seed=73 的工程 split。可以复用正式 split API，但输出目录必须与未来正式实验隔离。记录 train/val/test 的索引和哈希；本步骤只能访问 train/val。不要根据任何结果改变 split。
3. 依次运行两个极小真实 smoke，每个只训练 1 epoch，不做超参数搜索：
   - A：`random_k + TFPP full`，`k=1000`；用于验证真实 NPY mmap、训练折内众数、Deep/Wide 输入和GPU训练。
   - B：`GWAS→DC-SIS + TFPP full`，`gwas_k=5000`、`k=1000`；用于验证在2,860,770列上执行真实训练折内监督筛选及显式串联。
   - 如果生产 GWAS 包含 PCA 群体结构校正，必须确认 PCA 的中心化、缺失填补、方差过滤和分解只使用 train 行；marker panel 的选取不得读取 phenotype，panel size、PC 数、seed 和算法版本必须显式记录并进入缓存/产物哈希。改变 val/test 基因型或 phenotype 不得改变 PCA、GWAS 评分或筛选索引。若 PCA 是本步骤新增的算法变化，必须先停止并报告，不能作为工程 smoke 顺带静默加入。
4. 使用单一性状 RGBVI 和14个 DAS 时间点。模型配置使用当前安全默认值；batch size 从现有配置开始，若发生真实 OOM 可以下调并记录，但不得改模型结构。
5. 筛选和加载必须使用 mmap/分块实现，不得在内存或磁盘生成完整 SNP 副本。记录每阶段墙钟时间、峰值 CPU RSS、GPU 峰值显存、筛选缓存大小、checkpoint 大小、输出目录大小和剩余磁盘。
6. smoke 的 val loss 只用于确认数值有限、前向反向和 checkpoint 保存正常，不得报告为论文性能，不得据此选择筛选方法、k、模型结构或正式超参数。
7. 训练后通过严格加载器重新加载最佳 checkpoint，在 val 上复现预测；允许浮点容差必须预先固定并记录。仍不得访问 test。
8. 输出目录使用 `outputs/engineering_pretest_smoke/<config_hash>/`，禁止被正式汇总器纳入；manifest、status 和 README 明确写入 `paper_eligible=false`。正式 summarize 对该目录必须拒绝。
9. 增加合成回归测试，证明 pretest 模式不调用 test loader/phenotype/metrics，生产训练函数被实际复用，工程 smoke 不能进入正式汇总，重复配置禁止覆盖。
10. 开始真实运行前先执行 dry-run，展示数据 manifest、split、两个配置、预计输出目录和磁盘检查。若可用磁盘不足以保留原始数据之外的本次少量产物，停止并报告；不得删除用户文件。

完成后停止。不要运行5折、50000 SNP、ConvAE或正式实验，不要提交或 push。汇报两个 smoke 的成功/失败阶段、train/val 数量、实际筛选数、各阶段时间、峰值内存/显存、产物大小、val 数值是否有限、严格重载复现、test 未访问证据、测试结果和建议 commit message。不要汇报 test 指标。
```

通过标准：至少 random_k 路线在真实数据上完成 train/val 单轮闭环，且代码证据表明整个 smoke 没有读取 test；串联筛选若受性能限制，应给出准确瓶颈而不是扩大运行范围。

---

## 第 14 步：50,000 位点与正式 GWAS 路径容量验证

```text
请在真实数据上进行第二阶段 pre-test capacity smoke，验证计划中的 50,000 位点 TFPP 输入以及正式 runner 的完整 GWAS PCA 路径能否运行。本步骤仍然只使用 train/val，不得读取 test phenotype、生成 test 预测或 test 指标；所有产物标记 `paper_eligible=false`、`purpose=engineering_capacity_smoke`。

前提：
1. 当前 HEAD 必须是第 13 步提交，工作区干净，完整测试通过。
2. 复用第 13 步完全相同的真实 manifest、RGBVI、DAS 时间坐标、seed=73 和 train/val/test=320/40/40 索引哈希。不得重新随机划分。
3. 缺失策略继续显式使用训练折内 `mode`，test 保持完全未访问。

需要运行的两个任务：
1. `random_k + TFPP full`：`k=50000`，只训练 1 epoch。用于测量50,000输入维度下的数据加载、GPU显存、前向反向、checkpoint和val严格重载。
2. `GWAS→DC-SIS + TFPP full`：暂定 `gwas_k=200000`、最终 `k=50000`，只训练 1 epoch。GWAS必须调用正式生产 runner 使用的 `gwas_screen()` 定义，包括在全部训练矩阵 SNP 上拟合既有5-PC PCA，然后分块扫描全部2,860,770列；DC-SIS必须只消费实际保存的200,000个GWAS候选。不得调用或回退到第13步的20,000-marker pretest panel方法。

执行要求：
1. 先 dry-run，打印精确函数调用链、配置哈希、数据/split哈希、预计输出路径和磁盘余量。明确证明任务2使用生产 `gwas_screen()`；若现有 pretest入口不能安全调用生产路径，只增加薄的调度层，不复制筛选算法。
2. 两个任务依次运行，不能并行争抢内存/GPU。运行过程中记录阶段进度和墙钟时间，至少区分：读取/索引、PCA准备、PCA拟合、全量GWAS扫描、DC-SIS、模型构建、训练、验证、checkpoint重载。
3. 记录峰值 CPU RSS、GPU显存、临时文件、缓存、checkpoint和总目录大小。不得生成完整 SNP 矩阵副本。可使用内存映射、索引视图和现有分块接口。
4. 如果生产 `gwas_screen()` 当前必须物化训练矩阵，允许其在现有约755 GiB RAM内按原定义运行并准确记录峰值；不得为了跑通静默改成20,000 panel。若时间或内存不可接受，安全停止并报告瓶颈，再决定是否开发与原定义等价的分块实现。
5. 50,000位点训练先使用当前正式 batch size。只有真实 CUDA OOM 时才允许降低 batch size并重跑，必须保留失败日志和记录最终值；不得改模型宽度、层数或其他架构参数。
6. val loss只检查有限性。严格加载最佳 checkpoint并复现val预测，预先使用 `rtol=1e-5, atol=1e-6`。不得根据val数值调整 k、gwas_k 或模型参数。
7. 输出到 `outputs/engineering_capacity_smoke/<config_hash>/`，正式汇总器必须拒绝。不得覆盖第13步产物。
8. 增加必要的合成测试，证明 capacity模式不访问test、任务2拒绝pretest panel路径、调用生产GWAS、50,000维配置和实际筛选数量进入manifest/checkpoint、失败不会自动回退算法。

完成后停止。不要运行5折、ConvAE、消融或正式实验，不要提交或push。汇报两个任务的成功状态、实际筛选链、各阶段时间、峰值内存/显存、输出大小、val有限性和重载复现、test未访问证据、完整测试结果，以及对正式实验可行性的判断和建议commit message。不要汇报test指标。
```

通过标准：50,000位点TFPP完成单轮train/val闭环；正式全训练矩阵PCA + GWAS→DC-SIS路径完成或给出可定位的性能瓶颈，且不存在向工程panel算法的静默回退。

---

## 每一步检查无误后的提交提示词

```text
我已经检查并接受本步骤的修改。请再次运行本步骤相关测试；确认通过后，创建一个只包含本步骤改动的本地 Git commit，提交信息使用清晰的 conventional commit 格式。不要合并无关修改，不要 force，不要 push。完成后告诉我 commit SHA、包含的文件和测试结果。
```

推送前应先确认远端没有新的他人修改；如有更新，先审查差异并安全整合，不要强推覆盖。
