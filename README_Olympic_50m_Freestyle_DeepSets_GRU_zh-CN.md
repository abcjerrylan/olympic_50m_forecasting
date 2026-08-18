# 奥运会50米自由泳成绩预测

## DeepSets + GRU + 查询解码器实现规范（中文版）

本文件是 `README_Olympic_50m_Freestyle_DeepSets_GRU.md` 的中文版本。若两者出现歧义，以英文原始规范和实际数据审计结果为准。

## 1. 项目目标

仅使用世界游泳锦标赛历史成绩作为模型输入，预测奥运会男子、女子50米自由泳成绩。模型不预测运动员身份，姓名、国籍、泳道等身份信息不得进入模型特征。

每个奥运年份、每个性别输出27个时间：

- 进入半决赛的16名运动员在预赛中的成绩：`H1–H16`；
- 进入决赛的8名运动员在半决赛中的成绩：`S1–S8`；
- 决赛前三名的成绩：`F1–F3`。

每组均从快到慢排列，即 `H1 <= ... <= H16`、`S1 <= ... <= S8`、`F1 <= F2 <= F3`。

## 2. 输入数据

项目使用 `data` 目录中的两个工作簿：

```text
data/World_Aquatics_Championships_50m_Freestyle_Complete_Results_1986-2025.xlsx
data/Olympic_Games_50m_Freestyle_Complete_Results_1988-2024.xlsx
```

读取工作表 `All_Data`，预期字段为：

```text
Year, Edition, Name, Sex, Phase, Time_seconds, Time_raw, Status,
NAT, Heat, Lane, Rank, Official_event_page, Official_API_source
```

世锦赛数据用于训练和历史上下文；奥运数据只用于构造外部评估标签。预测奥运年份 `Y` 时，只允许使用 `Year < Y` 的世锦赛，严禁使用同年或未来的世锦赛，也不得将奥运成绩放入训练输入。

## 3. 奥运预测目标

### 3.1 预赛晋级成绩

找出实际参加半决赛的16名运动员，再回查他们在同届预赛中的成绩，按时间排序形成 `H1–H16`。姓名只允许作为预处理阶段的临时关联键，张量构造前必须删除。

若因记录缺失无法重建晋级链，允许回退为最快16个有效预赛成绩，但必须设置数据质量标记。

### 3.2 半决赛晋级成绩

找出实际参加决赛的8名运动员，回查其半决赛成绩，排序形成 `S1–S8`。无法重建时可回退为最快8个有效半决赛成绩，并记录标记。

### 3.3 决赛奖牌成绩

使用官方决赛名次确定奖牌获得者。并列导致官方名次为 `1、1、3` 等情况时，应保留原始官方名次元数据，并按确定性规则形成固定的三个输出位置。

### 3.4 早期赛制

1988、1992、1996年奥运会没有半决赛，采用预赛加A/B决赛赛制。这些届次可用于决赛和奖牌目标分析，但不能伪装成完整27输出样本；缺失阶段必须用 mask 表示。完整三阶段评估从2000年开始。

## 4. 使用全部世锦赛成绩

所有有效的预赛、预赛加赛、半决赛、半决赛加赛和决赛数值成绩都必须进入历史输入，不能只保留前16、前8或前3名。

有效数值记录应满足：

```text
Time_seconds 存在
Status == OK
时间为有限值且位于合理区间
```

不得静默删除异常状态。每届、每个性别、每个阶段记录以下计数：

```text
DNS_count, DSQ_count, DNF_count, missing_time_count, valid_result_count
```

加赛成绩作为独立子阶段保留，并通过 `subphase_id`、`is_swim_off` 区分。重复检查键建议使用 `Year, Sex, Phase, Name, Time_raw, Heat`，所有删除的重复记录都要写入审计日志。

## 5. 成绩 token

每条有效成绩转换为一个不含身份信息的 token：

```text
normalized_time
phase_rank_normalized
field_size_normalized
subphase_id
is_swim_off
```

其中：

```text
phase_rank_normalized = (phase_rank - 1) / max(field_size - 1, 1)
field_size_normalized = field_size / training_max_field_size
```

名次缺失或不一致时，按同届、同性别、同阶段的有效时间重新计算，并列使用竞赛排名。姓名、国籍、泳道和官方URL不得进入模型。

## 6. 归一化

所有统计量只能在当前训练折上拟合。男子/女子与预赛/半决赛/决赛分别使用独立时间缩放器：

```text
normalized_time = (time_seconds - training_mean) / training_std
```

也可配置为中位数与IQR的稳健缩放。年份特征包括：

```text
year_normalized = (year - base_year) / 10
target_year_gap = target_year - latest_context_year
edition_year_gap = current_year - previous_year
```

奥运测试成绩不得参与缩放器拟合。

## 7. 分层输入结构

每届世锦赛、每个性别包含三个可变长度集合：

```text
heats_tokens       [N_heats, token_dim]
semifinal_tokens   [N_semifinals, token_dim]
final_tokens       [N_finals, token_dim]
```

批处理中可以 padding，但 pooling 必须使用 mask 排除填充值。缺失半决赛应表示为空集合并设置 `semifinal_present = 0`，不得加入看似真实的零时间 token。

## 8. 模型结构

### 8.1 阶段专属 DeepSets

预赛、半决赛、决赛各使用一套独立编码器。token MLP：

```text
token_dim -> Linear(32) -> GELU -> Linear(32) -> GELU
```

对每个集合计算 masked mean、masked max、masked standard deviation，拼接为96维，再压缩为48维：

```text
Linear(96, 48) -> GELU -> Dropout(0.10)
```

### 8.2 届次编码器

拼接三个48维阶段表示、3维阶段存在标记、状态计数特征、4维性别嵌入、届次年份间隔和标准化年份，经过：

```text
Linear(input_dim, 96) -> GELU -> Dropout(0.10)
Linear(96, 64) -> GELU
```

得到64维届次表示。

### 8.3 因果 GRU

按时间顺序把历届世锦赛表示输入单向GRU：

```text
input_size=64, hidden_size=48, num_layers=1,
dropout=0, bidirectional=false
```

默认使用目标年份之前的全部世锦赛；可用 `max_context_editions` 做消融。GRU必须是因果的。

### 8.4 查询表示与解码器

每个查询包含目标年份、目标年份间隔、性别嵌入、目标阶段嵌入、绝对名次和 `log1p(rank)`。奥运推理查询为预赛1–16、半决赛1–8、决赛1–3。

最终GRU上下文与查询特征拼接后输入：

```text
Linear(64) -> GELU -> Dropout(0.10)
Linear(32) -> GELU -> Linear(1)
```

输出反归一化为秒。每个16、8、3结果组分别执行 isotonic projection，保存原始预测与投影预测，保证名次越后时间不更快。

模型总参数量应控制在约2万至8万，首版不使用大型 Transformer 或多层循环网络。

## 9. 训练查询

对目标世锦赛届次 `t`：

```text
Context = t之前的全部世锦赛
Query   = 目标年份、性别、阶段、名次
Label   = t届该成绩的官方时间
```

每个有效结果都形成监督查询。同一目标届次内的查询高度相关，必须保留在同一个训练或验证单元中，禁止随机拆分到不同集合。最少历史上下文默认三届。

## 10. 损失函数

基础损失采用 Huber。普通结果权重1.0；预赛前16权重2.0；半决赛前8权重2.5；决赛前3权重3.0；第16、第8、第3名截止线额外加1.0。

同届、同性别、同阶段的预测加入顺序惩罚：

```text
order_loss = mean(relu(predicted_time[r] - predicted_time[r+1]))
total_loss = weighted_huber_loss + lambda_order * order_loss
lambda_order = 2.0
```

## 11. 初始训练配置

```yaml
seed: 42
optimizer: AdamW
learning_rate: 0.001
weight_decay: 0.0001
max_epochs: 500
early_stopping_patience: 40
gradient_clip_norm: 1.0
dropout: 0.10
batch_unit: target_edition
min_context_editions: 3
max_context_editions: null
lambda_order: 2.0
device: auto
```

必须固定 Python、NumPy、PyTorch 随机种子，并保存配置、训练年份、验证年份、缩放参数、最佳轮次、模型和优化器状态。

## 12. 时间验证与奥运评估

禁止随机行级拆分。内部验证使用扩展窗口：较早世锦赛训练、下一届世锦赛验证。

历史奥运走步评估对每个年份独立执行：选择 `Year < Y` 的世锦赛、仅用这些数据拟合预处理和训练模型，再预测并对比奥运目标。

完整赛制评估年份为：

```text
2000, 2004, 2008, 2012, 2016, 2020, 2024
```

东京奥运仍使用官方届次年份2020。建议2000–2020用于开发诊断，2024作为锁定最终测试；一旦用2024调参，就不能再称其为未触碰测试集。

未来预测命令应接受目标年份，如2028，并使用此前全部可用世锦赛。若目标年份前仍可能新增世锦赛数据，输出必须标为 provisional。

## 13. 评估指标与输出

男女分别报告并提供合并汇总：

```text
MAE_Heats16, RMSE_Heats16
MAE_Semifinals8, RMSE_Semifinals8
MAE_Finals3, RMSE_Finals3
MAE_All27, RMSE_All27
```

截止线指标：预赛第16名、半决赛第8名、决赛第3名、决赛第1名绝对误差。另报告 `mean(predicted - actual)` 偏差；正值表示预测更慢。

预测CSV至少包含：

```text
target_year, sex, target_group, rank,
predicted_time_seconds, actual_time_seconds,
absolute_error_seconds, signed_error_seconds,
is_cutoff, training_year_max, run_id
```

未来预测的实际值和误差字段留空。

## 14. 可视化

男女分别生成：世锦赛阶段分布、奥运预赛16人实际与预测、半决赛8人实际与预测、决赛前三实际与预测、三条截止线趋势、逐届奥运MAE、预测与实际散点图及 `y=x` 参考线。横轴使用比赛年份，纵轴为秒。

## 15. 项目结构

```text
olympic_50m_forecasting/
├── README.md
├── README_Olympic_50m_Freestyle_DeepSets_GRU.md
├── README_Olympic_50m_Freestyle_DeepSets_GRU_zh-CN.md
├── requirements.txt
├── configs/base.yaml
├── data/
│   ├── 两个原始Excel
│   └── processed/
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── target_builder.py
│   ├── datasets.py
│   ├── losses.py
│   ├── metrics.py
│   ├── train_utils.py
│   ├── visualization.py
│   └── models/
├── scripts/
├── tests/
└── outputs/runs/
```

## 16. 命令行

在项目根目录运行：

```powershell
python scripts/prepare_data.py
python scripts/train.py --config configs/base.yaml --output-dir outputs/runs/main
python scripts/evaluate_olympics.py --config configs/base.yaml --years 2000 2004 2008 2012 2016 2020 2024 --output-dir outputs/runs/walk_forward
python scripts/predict_olympics.py --checkpoint outputs/runs/main/best_model.pt --target-year 2028 --output outputs/runs/main/olympics_2028_predictions.csv
pytest -q
```

## 17. 数据泄漏要求

自动测试必须保证：目标年份不使用同年或未来世锦赛；奥运数据不进入训练；缩放器只在训练折拟合；同届查询不被拆分；姓名和国籍不进入张量；padding不影响pooling；缺失半决赛保持mask。违规时应立即抛出明确异常。

## 18. 基线与消融

必须实现“最近一届世锦赛中相同性别、阶段和名次的成绩”基线。消融包括：去掉GRU、只用头部晋级成绩、去掉状态计数、只用mean pooling。所有方案使用相同走步折。

## 19. 验收标准

两个Excel均能读取；全部有效世锦赛成绩进入历史集合；身份字段不进入张量；可变长度和缺失阶段正确处理；每个性别恰好输出27项；组内单调；历史奥运只使用更早世锦赛；指标、CSV和图表完整；训练可复现；pytest通过；CPU冒烟训练成功。

关键形状：

```text
阶段编码器        [batch, edition, 48]
届次编码器        [batch, edition, 64]
GRU上下文         [batch, 48]
查询解码器        [number_of_queries, 1]
单性别奥运输出    [27]
双性别奥运输出    [2, 27]
```

## 20. 已知限制

独立比赛届次数远少于成绩记录数，同届结果高度相关。模型应保持小型化，并依赖正则化、早停、届次级验证和不确定性报告。其他限制包括泳衣规则、计时技术、晋级制度、早期缺少半决赛、世锦赛间隔不规则、东京2020实际在2021举行、奥运与世锦赛条件差异等。低训练损失不能证明未来奥运预测准确。

## 21. 可选不确定性

确定性模型稳定后，可使用种子 `[11, 22, 33, 44, 55]` 训练5模型集成，报告均值、标准差和经验95%区间。在数据管道、泄漏测试和确定性模型验证完成前，不应优先实现该功能。

## 22. 推荐实现顺序

1. 建立项目与配置；
2. 加载并验证两个Excel；
3. 构造奥运晋级目标；
4. 构造世锦赛可变长度集合；
5. 实现训练折归一化；
6. 实现masked DeepSets；
7. 实现届次编码器与因果GRU；
8. 实现查询与解码器；
9. 实现加权Huber与顺序损失；
10. 实现扩展窗口训练和奥运走步评估；
11. 生成指标、CSV和图表；
12. 完成泄漏测试与CPU冒烟训练。

## 23. 最终模型定义

最终系统是一个分层神经预测模型：使用阶段专属DeepSets编码每届世锦赛男子、女子50米自由泳预赛、半决赛、决赛的全部有效成绩；使用因果GRU建模比赛随时间的发展；使用条件查询解码器预测奥运预赛晋级、半决赛晋级和决赛前三成绩。奥运数据仅用于时间隔离评估，模型不使用任何运动员身份信息。
