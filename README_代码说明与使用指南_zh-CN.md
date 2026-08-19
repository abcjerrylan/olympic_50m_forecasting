# 代码说明与使用指南

这份文档说明本项目究竟解决什么问题、数据如何流动、每个代码文件负责什么，以及如何使用 NVIDIA GPU 完成训练、评估和未来预测。

## 1. 项目最终做什么

项目用历届世界游泳锦标赛男子、女子50米自由泳的完整成绩分布，预测某届奥运会的三个结果组：

```text
预赛晋级者的预赛成绩       16个：H1-H16
决赛晋级者的半决赛成绩      8个：S1-S8
决赛奖牌获得者成绩          3个：F1-F3
每个性别合计                   27个
男女合计                       54个
```

它预测的是“成绩分布和名次位置”，不是运动员姓名。运动员姓名只在构造奥运晋级标签时短暂用于跨轮次关联，不会进入模型张量。

## 2. 两个 Excel 分别有什么作用

### 世锦赛工作簿

```text
data/World_Aquatics_Championships_50m_Freestyle_Complete_Results_1986-2025.xlsx
```

这是唯一训练来源，也是预测某届奥运时的历史上下文。预测年份 `Y` 时，只能使用 `Year < Y` 的世锦赛。

### 奥运会工作簿

```text
data/Olympic_Games_50m_Freestyle_Complete_Results_1988-2024.xlsx
```

它只用于构造真实答案和计算历史评估误差，不会加入训练输入。

## 3. 从 Excel 到预测结果的完整流程

```text
Excel All_Data
    |
    v
字段、状态、时间、重复记录校验
    |
    +--> 数据质量报告
    |
    v
每届、每性别、每阶段的可变长度成绩集合
    |
    v
阶段专属 DeepSets 编码器
    |
    v
每届比赛的64维向量
    |
    v
按年份排列的单向因果 GRU
    |
    v
目标年份 + 性别 + 阶段 + 名次查询
    |
    v
Query Decoder 输出一个标准化时间
    |
    v
反归一化为秒 + 单调投影
    |
    v
CSV、指标、图表、模型检查点
```

## 4. 数据预处理具体做什么

### 4.1 加载和校验

`src/data_loader.py`：

- 检查 `All_Data` 工作表及14个必要字段；
- 将年份、时间、性别、阶段和状态转换为统一格式；
- 判断成绩是否满足 `Status == OK`、时间有限且处于合理范围；
- 区分主阶段和 swim-off 子阶段；
- 检测重复记录并生成质量信息；
- 提供项目内两个原始工作簿的默认路径。

### 4.2 构造奥运真实标签

`src/target_builder.py`：

- 找出实际参加半决赛的运动员，回查其预赛成绩，得到16个预赛晋级时间；
- 找出实际参加决赛的运动员，回查其半决赛成绩，得到8个半决赛晋级时间；
- 按官方名次提取三位奖牌获得者；
- 正确处理2000年男子 `1、1、3` 的并列奖牌名次；
- 对早期A/B决赛只使用 Final A 构造奖牌目标；
- 姓名在函数结束后不再保留为模型特征。

### 4.3 构造模型输入

`src/preprocessing.py`：

每条有效成绩转成5维 token：

```text
1. normalized_time          训练折标准化后的时间
2. phase_rank_normalized    该阶段内部的相对名次
3. field_size_normalized    参赛规模相对训练最大规模
4. subphase_id              主阶段或加赛
5. is_swim_off              是否为加赛
```

同时为每届比赛构造：

- 三个阶段存在标记；
- DNS、DSQ、DNF、缺失时间、有效成绩数量；
- 标准化年份；
- 与上一届的年份间隔；
- 性别编号。

时间缩放器按“男子/女子 × 预赛/半决赛/决赛”分别拟合，而且只能查看当前训练折。

## 5. 模型结构

### 5.1 DeepSets：读取一届比赛的完整分布

文件：`src/models/deepsets.py`

一届比赛可能有几十到两百条预赛成绩，但半决赛通常只有16条，决赛通常只有8条。因此不能使用固定长度普通全连接网络。

DeepSets 对集合中的每条成绩使用同一个 MLP：

```text
5 -> Linear(32) -> GELU -> Linear(32) -> GELU
```

然后对整组成绩计算：

```text
masked mean + masked max + masked standard deviation
```

三个32维统计量拼接成96维，再压缩为48维。预赛、半决赛、决赛分别使用独立编码器，因为三个阶段的成绩分布含义不同。

`masked` 的作用是让 padding 和缺失阶段不会影响统计结果。

### 5.2 届次编码器

文件：`src/models/full_model.py`

把三个48维阶段表示、阶段是否存在、状态计数、性别嵌入和年份特征拼接，再转换成64维的“这一届世锦赛总结向量”。

### 5.3 因果 GRU：建模年代变化

文件：`src/models/temporal_gru.py`

将历届世锦赛向量按年份从早到晚输入单层、单向GRU。GRU最后的48维隐藏状态代表截至目标年份之前的历史发展趋势。

它不能看到目标年份或之后的世锦赛，因此是因果模型。

### 5.4 Query Decoder：按阶段和名次提问

文件：`src/models/query_decoder.py`

模型不是一次固定输出27个值，而是逐项提问。例如：

```text
目标年份=2028，性别=Male，阶段=HEATS，名次=16
```

查询特征包括目标年份、距最近历史届次的间隔、性别、阶段、绝对名次和名次对数。解码器把查询与GRU上下文拼接后输出一个时间。

推理时发出16+8+3个查询。最后使用 isotonic projection 保证同组成绩从快到慢排列。

当前完整模型共有约66,677个可训练参数，属于适合小届次数数据的紧凑模型。

## 6. 模型如何训练

训练样本不是随机拆分的运动员成绩，而是以“目标世锦赛届次 + 性别”为单位。

例子：

```text
上下文：1986到2017的全部世锦赛
目标届次：2019
查询：2019男子预赛第37名
标签：2019男子预赛第37名的真实时间
```

同一个目标届次的全部查询放在同一个优化步骤中，以便计算组内顺序损失。

损失由两部分组成：

```text
加权 Huber 时间误差
+ lambda_order * 相邻名次顺序惩罚
```

预赛前16、半决赛前8、决赛前3以及关键截止名次权重更高。

`src/losses.py` 实现损失，`src/train_utils.py` 负责随机种子、设备选择、训练循环、早停、梯度裁剪、检查点和预测。

## 7. GPU 是怎样被使用的

配置文件现在默认为：

```yaml
device: cuda
allow_tf32: true
```

训练开始时会：

1. 检查当前 PyTorch 是否真的支持 CUDA；
2. 选择 `cuda:0`；
3. 将模型参数移动到显卡；
4. 将每个届次的所有 token、查询和标签移动到显卡；
5. 在显卡上执行前向传播、反向传播和 AdamW 更新；
6. 在 `run_summary.json` 中记录显卡型号、CUDA构建和峰值显存。

如果 CUDA 不可用，配置为 `cuda` 时程序会直接报出安装提示，不再悄悄退回 CPU。

这个模型只有约6.7万参数，数据届次也很少，因此即使使用RTX 5070，GPU利用率和显存占用仍可能很低；这是模型规模造成的，不代表程序没有使用GPU。对本项目而言，GPU主要缩短多折、消融和多随机种子实验的总时间。

## 8. 环境安装

建议在项目专用 Conda 环境中操作，避免污染 `base`：

```powershell
conda create -n swimming-forecast python=3.13 -y
conda activate swimming-forecast
```

RTX 5070 建议安装官方 CUDA 13.0 版 PyTorch：

```powershell
python -m pip install torch==2.13.0+cu130 --index-url https://download.pytorch.org/whl/cu130
python -m pip install -r requirements.txt
```

验证：

```powershell
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

预期至少看到：

```text
2.13.0+cu130
13.0
True
NVIDIA GeForce RTX 5070 ...
```

## 9. 推荐使用顺序

所有命令都应在项目根目录运行：

```powershell
cd D:\swimming_prediction\olympic_50m_forecasting
```

### 9.1 准备数据

```powershell
python scripts/prepare_data.py
```

生成：

```text
data/processed/world_championships.pkl
data/processed/olympics.pkl
data/processed/olympic_targets.csv
data/processed/olympic_target_quality.csv
data/processed/data_quality.json
```

### 9.2 运行测试

```powershell
python -m pytest tests -q -p no:cacheprovider
```

### 9.3 GPU短训练

先用少量epoch确认整条链路：

```powershell
python scripts/train.py `
  --config configs/base.yaml `
  --device cuda `
  --max-epochs 5 `
  --patience 3 `
  --output-dir outputs/runs/gpu_smoke
```

启动日志必须显示 `Compute device: cuda:0` 和 RTX 5070 的名称。

### 9.4 主训练

```powershell
python scripts/train.py `
  --config configs/base.yaml `
  --device cuda `
  --output-dir outputs/runs/main_gpu
```

主要输出：

```text
best_model.pt          最佳模型、优化器、缩放器、配置和设备信息
training_history.csv   每个epoch的训练与验证损失
run_summary.json       最佳epoch、参数量、GPU和峰值显存
data_quality.json      原始数据质量摘要
```

### 9.5 历史奥运走步评估

当前早期赛制存在真实冷启动边界：2000年前的世锦赛没有半决赛；2004折的当前验证划分也会使缩放训练段缺少半决赛。未经冷启动策略修改前，稳定命令从2008开始：

```powershell
python scripts/evaluate_olympics.py `
  --config configs/base.yaml `
  --device cuda `
  --years 2008 2012 2016 2020 2024 `
  --output-dir outputs/runs/walk_forward_gpu
```

每个奥运年份会独立训练一个模型，因此这一步最适合使用GPU。

输出包括逐项预测、要求指标、截止线指标、最近世锦赛基线和图表。

### 9.6 预测2028奥运会

```powershell
python scripts/predict_olympics.py `
  --checkpoint outputs/runs/main_gpu/best_model.pt `
  --device cuda `
  --target-year 2028 `
  --output outputs/runs/main_gpu/olympics_2028_predictions.csv
```

未来预测没有真实成绩和误差字段，并标记为 provisional。

## 10. 常用参数

```text
--device cuda             强制使用第一张GPU
--device cpu              明确使用CPU
--max-epochs N            最大训练轮数
--patience N              早停耐心轮数
--validation-year YEAR    主训练验证世锦赛年份
--no-gru                  消融：不使用GRU
--mean-pooling            消融：只使用mean pooling
--no-count-features       消融：不使用状态计数
```

## 11. 如何确认真的在使用 GPU

方法一：查看训练启动日志，应包含：

```text
Compute device: cuda:0
GPU: NVIDIA GeForce RTX 5070 ...
```

方法二：另开终端运行：

```powershell
nvidia-smi -l 1
```

方法三：训练后查看：

```text
outputs/runs/<run_name>/run_summary.json
```

其中记录 `device_info.cuda_device_name` 和 `peak_cuda_memory_mb`。

## 12. 主要代码文件索引

```text
src/data_loader.py             Excel读取、字段校验、状态和重复检查
src/preprocessing.py           缩放器、token、上下文、训练查询
src/target_builder.py          奥运晋级链和27项真实标签
src/datasets.py                NumPy数据转PyTorch张量并移动到设备
src/models/deepsets.py         阶段集合编码器和masked pooling
src/models/temporal_gru.py     历史届次GRU
src/models/query_decoder.py    条件查询解码器
src/models/full_model.py       完整模型组装
src/losses.py                  加权Huber与顺序损失
src/metrics.py                 MAE、RMSE、偏差、截止线指标
src/train_utils.py             GPU、训练循环、早停、保存和预测
src/visualization.py           分布、误差和预测图表
scripts/prepare_data.py        数据准备入口
scripts/train.py               主模型训练入口
scripts/evaluate_olympics.py   历史奥运走步评估入口
scripts/predict_olympics.py    未来奥运预测入口
tests/                         泄漏、目标、mask和形状测试
```

## 13. 如何理解模型结果

训练损失低不代表未来奥运预测一定准确。有效结论应优先查看：

- 锁定年份的 `required_metrics.csv`；
- 与 `baseline_required_metrics.csv` 的比较；
- 男、女和三个阶段是否表现一致；
- 截止名次误差；
- 多个历史奥运年份上的稳定性，而不是单一届次。

如果神经网络没有超过“最近一届世锦赛同名次”基线，应如实报告，并进一步检查小样本、阶段制度变化和过拟合，而不能只展示训练损失。
