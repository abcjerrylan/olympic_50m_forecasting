# Olympic 50 m Freestyle Result Forecasting

## DeepSets + GRU + Query Decoder Implementation Specification

## 1. Project objective

Build a neural-network system that uses **World Aquatics Championships results only** as historical model input and predicts Olympic Games men's and women's 50 m freestyle results.

The model does **not** predict athlete identities. Athlete name, nationality, lane and other identity-related information must not be used as model features.

For every target Olympic edition and sex, the system must predict 27 times:

1. The heat times of the 16 swimmers who advance to the semifinals.
2. The semifinal times of the 8 swimmers who advance to the final.
3. The final times of the top 3 finishers.

The output for one sex is therefore:

```text
Heats qualifiers:      H1, H2, ..., H16
Semifinal qualifiers:  S1, S2, ..., S8
Final medalists:       F1, F2, F3
Total:                 27 predicted times
```

All results within each output group must be ordered from fastest to slowest:

```text
H1 <= H2 <= ... <= H16
S1 <= S2 <= ... <= S8
F1 <= F2 <= F3
```

The initial implementation must use the **DeepSets + GRU + Query Decoder** architecture described below.

---

## 2. Input files

The project is based on two previously prepared Excel workbooks:

```text
World_Aquatics_Championships_50m_Freestyle_Complete_Results_1986-2025.xlsx
Olympic_Games_50m_Freestyle_Complete_Results_1988-2024.xlsx
```

Expected worksheet:

```text
All_Data
```

Expected columns:

```text
Year
Edition
Name
Sex
Phase
Time_seconds
Time_raw
Status
NAT
Heat
Lane
Rank
Official_event_page
Official_API_source
```

### Data-source roles

- The World Championships workbook is the training and historical-context dataset.
- The Olympic Games workbook is the external evaluation and test-label dataset.
- Olympic results must never be inserted into the historical input sequence.
- A prediction for Olympic year `Y` may use only World Championships editions with `Year < Y`.
- World Championships held after a target Olympics must never be used to predict that Olympics.

Example:

```text
2024 Olympic prediction:
Allowed World Championships: 1986 through 2023
Forbidden World Championships: 2024 and 2025
```

---

## 3. Exact prediction targets

The prediction target is based on official progression between rounds, not athlete identity prediction.

### 3.1 Heats-to-semifinals target

For each Olympic edition and sex:

1. Identify the 16 swimmers who officially competed in the semifinals.
2. Retrieve those swimmers' heat times from the same edition.
3. Sort the 16 heat times from fastest to slowest.
4. Store them as `H1` through `H16`.

This definition correctly handles ties and heat swim-offs. If names are needed to join a swimmer's heat and semifinal records during preprocessing, they may be used **only as temporary join keys**. Names must be deleted before model tensors are created and must never become model features.

If progression cannot be reconstructed because of missing records, use the fastest 16 valid heat times as a documented fallback and set a data-quality flag.

### 3.2 Semifinals-to-final target

For each Olympic edition and sex:

1. Identify the 8 swimmers who officially competed in the final.
2. Retrieve those swimmers' semifinal times.
3. Sort the 8 semifinal times from fastest to slowest.
4. Store them as `S1` through `S8`.

If progression cannot be reconstructed, use the fastest 8 valid semifinal times as a documented fallback.

### 3.3 Final top-three target

For each Olympic edition and sex:

1. Select final results with official ranks 1, 2 and 3.
2. Retain valid official final times.
3. Store them as `F1`, `F2` and `F3`.

Official rank takes precedence over sorting when medals are tied. If a tie produces more than three medalists, retain the official rank information in the processed metadata and document the rule used to form the fixed three-value evaluation vector.

### 3.4 Early editions without semifinals

The 1988, 1992 and 1996 Olympic 50 m freestyle formats did not use semifinals. Therefore:

- They may be used to evaluate final and medal-related targets where definitions are available.
- They must not be treated as complete 27-output evaluation examples.
- Full three-stage evaluation begins with the 2000 Olympic Games.
- Missing stages must be represented by masks, not fabricated values or zeros.

---

## 4. Use every collected World Championships result

Every valid numerical World Championships time from the following phases must enter the model input:

```text
Heats
Heat Swim-off
Semifinals
Semifinal Swim-off
Finals
```

Do not keep only the top 16, top 8 or top 3 as historical input. The complete valid result distribution must be encoded.

### Valid numerical records

Use a result as a numerical token when:

```text
Time_seconds is present
Status == OK
Time_seconds is finite and within a physically plausible range
```

Do not silently delete abnormal statuses. Encode the following as per-edition, per-sex, per-phase count features:

```text
DNS_count
DSQ_count
DNF_count
missing_time_count
valid_result_count
```

Swim-off results should be retained as separate subphase information. They may share the main phase encoder with an additional `subphase_id` feature.

### Duplicate protection

The source workbook is expected to contain phase-summary rows rather than duplicated summary plus individual-heat rows. Nevertheless, preprocessing must check for duplicates using a key such as:

```text
Year, Sex, Phase, Name, Time_raw, Heat
```

Log all detected and removed duplicates.

---

## 5. Result-token representation

Each valid result becomes one token. No identity field may be included.

Recommended token features:

```text
normalized_time
phase_rank_normalized
field_size_normalized
subphase_id
is_swim_off
```

Definitions:

```text
phase_rank_normalized = (phase_rank - 1) / max(field_size - 1, 1)
field_size_normalized = field_size / training_max_field_size
```

The rank must be recomputed from valid times within the edition, sex and phase when the official rank is unavailable or inconsistent. Use competition ranking for ties.

Do not include:

```text
Name
NAT
Lane
Official URLs
```

These columns may remain in audit tables but must not enter the model.

---

## 6. Normalization

All preprocessing statistics must be fitted on the training fold only.

Use separate time scalers for:

```text
Male + Heats
Male + Semifinals
Male + Finals
Female + Heats
Female + Semifinals
Female + Finals
```

Recommended transformation:

```text
normalized_time = (time_seconds - training_mean) / training_std
```

Alternative robust transformation is allowed:

```text
normalized_time = (time_seconds - training_median) / training_IQR
```

The selected method must be configured and recorded in each run.

Normalize years using training data only:

```text
year_normalized = (year - base_year) / 10
target_year_gap = target_year - latest_context_year
edition_year_gap = current_year - previous_year
```

Never fit or refit scalers using Olympic test results.

---

## 7. Hierarchical input structure

The data hierarchy is:

```text
Competition edition
└── Sex
    ├── Heats result set: variable length
    ├── Semifinals result set: variable length
    └── Finals result set: variable length
```

For each World Championships edition and sex, build three masked result sets:

```text
heats_tokens:       [N_heats, token_dim]
semifinal_tokens:   [N_semifinals, token_dim]
final_tokens:       [N_finals, token_dim]
```

The dataloader may pad sets inside a mini-batch, but padded tokens must be excluded by masks from every pooling operation.

For an edition without a semifinal phase:

```text
semifinal_tokens = empty set
semifinal_present = 0
```

Do not replace a missing phase with real-looking zero-time tokens.

---

## 8. Model architecture

### 8.1 Stage-specific DeepSets encoders

Use a separate encoder for each main phase because heats, semifinals and finals represent different competitive distributions.

For every result token `x_i`:

```text
token_dim -> Linear(32) -> GELU -> Linear(32) -> GELU
```

This shared token MLP is applied independently to every token within the stage.

Masked pooling must calculate:

```text
masked mean
masked maximum
masked standard deviation
```

Concatenate the three pooled vectors:

```text
32 mean + 32 max + 32 std = 96-dimensional stage embedding
```

Then compress:

```text
Linear(96, 48) -> GELU -> Dropout(0.10)
```

Recommended stage encoders:

```text
HeatsDeepSetEncoder
SemifinalsDeepSetEncoder
FinalsDeepSetEncoder
```

### 8.2 Edition encoder

Concatenate:

```text
heats_embedding                 48
semifinals_embedding           48
finals_embedding               48
phase-presence mask             3
status/count features           configurable
sex embedding                   4
edition year gap                1
normalized edition year         1
```

Pass the concatenated vector through:

```text
Linear(input_dim, 96)
GELU
Dropout(0.10)
Linear(96, 64)
GELU
```

The output is a 64-dimensional edition embedding.

### 8.3 Temporal GRU

Feed chronologically ordered World Championships edition embeddings into a GRU:

```text
input_size: 64
hidden_size: 48
num_layers: 1
dropout: 0 for a single GRU layer
bidirectional: false
```

The GRU must be causal. It may see only editions earlier than the target year.

Use all available earlier World Championships by default. Support an optional `max_context_editions` configuration for ablation experiments.

### 8.4 Query representation

The decoder predicts one time for one requested phase/rank query.

Query features:

```text
target_year_normalized
target_year_gap
sex_embedding
target_phase_embedding
target_rank
log1p(target_rank)
```

Target phase values:

```text
HEATS
SEMIFINALS
FINALS
```

Do not require a future phase field size as a query input. The future Olympic heat field size may be unknown, and using it would introduce an unnecessary assumption. Absolute rank, log-rank, phase and historical result-distribution context are sufficient for the initial model.

At Olympic inference time, issue the following queries:

```text
HEATS ranks 1 through 16
SEMIFINALS ranks 1 through 8
FINALS ranks 1 through 3
```

### 8.5 Query decoder

Concatenate the final GRU context with the query representation:

```text
[gru_context, query_features]
        -> Linear(64)
        -> GELU
        -> Dropout(0.10)
        -> Linear(32)
        -> GELU
        -> Linear(1)
```

The decoder output is one normalized predicted time, which must be inverse-transformed into seconds.

### 8.6 Monotonic output projection

Because the Query Decoder predicts one rank at a time, apply an isotonic projection separately to each 16-value, 8-value and 3-value output group after inverse transformation. Preserve both raw and projected predictions in the output files.

The projected predictions must satisfy:

```text
time(rank r) <= time(rank r+1)
```

### 8.7 Model size

Keep the first implementation approximately between 20,000 and 80,000 trainable parameters. Print the exact parameter count at startup.

Do not build a large Transformer or a multi-layer recurrent network for the initial version.

---

## 9. Training-query construction

To use all collected results as supervision, train the decoder on all valid results from the next World Championships edition.

For target World Championships edition `t`:

```text
Context = all World Championships editions before t
Query   = target year, sex, phase and rank for one result in edition t
Label   = that result's official time
```

Repeat the query for every valid heat, semifinal and final result in the target edition.

Example:

```text
Context: all World Championships through 2017
Query:   year=2019, sex=Male, phase=Heats, rank=37
Label:   2019 men's heats rank-37 time
```

This produces many supervised query rows while preserving the fact that the number of independent competition editions is small.

Important: queries from the same target edition are correlated. They must remain in the same split. Never randomly split individual result rows across training and validation.

Minimum historical context should default to three earlier World Championships editions.

---

## 10. Loss function

Use Huber loss for the primary time error:

```text
base_loss = Huber(predicted_time, actual_time)
```

Use query weights so that every result contributes, while the final Olympic targets receive greater emphasis during World Championships training:

```text
All other valid results:                   weight 1.0
Heats ranks 1-16:                          weight 2.0
Semifinals ranks 1-8:                      weight 2.5
Finals ranks 1-3:                          weight 3.0
Heats rank 16 cutoff:                      additional weight 1.0
Semifinals rank 8 cutoff:                  additional weight 1.0
Finals rank 3 medal cutoff:                additional weight 1.0
```

Add an ordering penalty for batches of queries from the same edition, sex and phase:

```text
order_loss = mean(relu(predicted_time[r] - predicted_time[r+1]))
```

Total loss:

```text
total_loss = weighted_huber_loss + lambda_order * order_loss
lambda_order default = 2.0
```

Because each decoder call produces one value, calculate ordering loss after grouping predictions belonging to the same target edition, sex and phase.

---

## 11. Training configuration

Recommended initial configuration:

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
loss: weighted_huber
lambda_order: 2.0
device: auto
```

Use deterministic random seeds for Python, NumPy and PyTorch. Save:

```text
random seed
configuration
training years
validation years
scaler parameters
best epoch
model state
optimizer state
```

Early stopping and hyperparameter selection must use World Championships validation data, not the final Olympic test edition.

---

## 12. Time-based validation and Olympic evaluation

Random row-level splitting is prohibited.

### 12.1 Internal World Championships validation

Use expanding-window validation:

```text
Train on earlier World Championships -> validate on the next World Championships
```

Example folds:

```text
Train through 2009 -> validate 2011
Train through 2011 -> validate 2013
Train through 2013 -> validate 2015
...
Train through 2023 -> validate 2024 or 2025 as configured
```

Use these folds for architecture and hyperparameter decisions.

### 12.2 Historical Olympic walk-forward evaluation

For each target Olympics `Y`:

1. Select World Championships editions with `Year < Y`.
2. Fit preprocessing scalers using only those World Championships.
3. Train the model using only those World Championships.
4. Predict 27 results for each sex.
5. Compare predictions with the corresponding Olympic targets.

Required complete-format Olympic folds:

```text
2000
2004
2008
2012
2016
2020
2024
```

Use official edition year `2020` for Tokyo 2020 even though the competition took place in 2021.

### 12.3 Locked final test

Recommended research split:

```text
Development and diagnostic Olympics: 2000-2020
Locked final test:                    2024
```

Do not tune the architecture or hyperparameters based on 2024 Olympic performance. If 2024 is used during development, it must no longer be described as an untouched test set.

### 12.4 Future Olympic prediction

The prediction command must accept a target year, for example `2028`.

For a 2028 prediction, use every available World Championships edition before 2028. Mark the output as provisional when future pre-2028 World Championships results are not yet available.

---

## 13. Evaluation metrics

Report metrics separately for men and women and as a combined summary.

Required group metrics:

```text
MAE_Heats16
RMSE_Heats16
MAE_Semifinals8
RMSE_Semifinals8
MAE_Finals3
RMSE_Finals3
MAE_All27
RMSE_All27
```

Required cutoff metrics:

```text
Heats_rank16_absolute_error
Semifinals_rank8_absolute_error
Finals_rank3_absolute_error
Finals_rank1_absolute_error
```

Also report bias:

```text
mean(predicted_time - actual_time)
```

Positive bias means the predicted result is slower than the actual result.

Every prediction CSV must contain:

```text
target_year
sex
target_group
rank
predicted_time_seconds
actual_time_seconds
absolute_error_seconds
signed_error_seconds
is_cutoff
training_year_max
run_id
```

For future Olympic predictions, actual and error fields remain blank.

---

## 14. Required visualizations

Use year as the horizontal axis and time in seconds as the vertical axis.

Generate separate figures for men and women.

Required plots:

1. Historical World Championships distributions by phase.
2. Olympic actual versus predicted heat qualifier times, ranks 1-16.
3. Olympic actual versus predicted semifinal qualifier times, ranks 1-8.
4. Olympic actual versus predicted final medal times, ranks 1-3.
5. Qualification cutoff trend for heat rank 16, semifinal rank 8 and final rank 3.
6. Per-Olympics MAE over time.
7. Predicted versus actual scatter plot with a `y=x` reference line.

Plotting convention:

```text
x-axis: competition year
y-axis: time in seconds
```

Optionally invert the y-axis so faster times appear higher, but label this clearly.

---

## 15. Required project structure

Codex should create a clean Python project similar to:

```text
olympic_50m_forecasting/
├── README.md
├── requirements.txt
├── configs/
│   └── base.yaml
├── data/
│   ├── raw/
│   ├── processed/
│   └── README.md
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── target_builder.py
│   ├── datasets.py
│   ├── losses.py
│   ├── metrics.py
│   ├── train_utils.py
│   ├── visualization.py
│   └── models/
│       ├── __init__.py
│       ├── deepsets.py
│       ├── temporal_gru.py
│       ├── query_decoder.py
│       └── full_model.py
├── scripts/
│   ├── prepare_data.py
│   ├── train.py
│   ├── evaluate_olympics.py
│   └── predict_olympics.py
├── tests/
│   ├── test_preprocessing.py
│   ├── test_no_future_leakage.py
│   ├── test_masked_pooling.py
│   ├── test_target_builder.py
│   └── test_model_shapes.py
└── outputs/
    └── runs/
```

Use PyTorch for the neural-network implementation.

Recommended dependencies:

```text
python>=3.11
torch
pandas
numpy
scikit-learn
openpyxl
pyyaml
matplotlib
seaborn
tqdm
pytest
```

---

## 16. Command-line interface

The exact argument names may be adjusted, but the project must provide equivalent commands.

### Prepare data

```bash
python scripts/prepare_data.py \
  --world-championships data/raw/World_Aquatics_Championships_50m_Freestyle_Complete_Results_1986-2025.xlsx \
  --olympics data/raw/Olympic_Games_50m_Freestyle_Complete_Results_1988-2024.xlsx \
  --output-dir data/processed
```

### Train

```bash
python scripts/train.py \
  --config configs/base.yaml \
  --world-championships data/processed/world_championships.pt \
  --output-dir outputs/runs/main
```

### Historical Olympic evaluation

```bash
python scripts/evaluate_olympics.py \
  --config configs/base.yaml \
  --world-championships data/processed/world_championships.pt \
  --olympics data/processed/olympics.pt \
  --years 2000 2004 2008 2012 2016 2020 2024 \
  --output-dir outputs/runs/walk_forward
```

### Predict a future Olympics

```bash
python scripts/predict_olympics.py \
  --checkpoint outputs/runs/main/best_model.pt \
  --world-championships data/processed/world_championships.pt \
  --target-year 2028 \
  --output outputs/runs/main/olympics_2028_predictions.csv
```

---

## 17. Data-leakage requirements

The implementation must include automated tests for the following rules:

1. A target year may not use a World Championships edition from the same year or a later year.
2. Olympic results may not appear in training inputs.
3. Scalers must be fitted using the training fold only.
4. Results from one target edition may not be divided between training and validation.
5. Names and nationality codes may not appear in model tensors.
6. Padded results may not affect DeepSets pooling.
7. Missing semifinal phases must remain masked.

Fail fast with a clear exception when any leakage rule is violated.

---

## 18. Required baselines and ablations

Although the final model is DeepSets + GRU + Query Decoder, implement simple references to verify that the neural network adds value.

Required non-neural baseline:

```text
Use the latest available World Championships result at the same sex, phase and rank.
```

Required ablations:

1. DeepSets + Query Decoder without GRU.
2. GRU using only top qualifiers instead of all results.
3. Full model without status/count features.
4. Full model with mean pooling only.

The full model should be compared against these variants using identical walk-forward folds.

---

## 19. Tests and acceptance criteria

The implementation is complete only when all of the following are satisfied:

- Both Excel files load successfully from `All_Data`.
- All valid World Championships heat, semifinal and final times are included in historical input sets.
- No athlete identity information enters model tensors.
- The model supports different numbers of results in different editions.
- Missing stages are handled with masks.
- The model outputs exactly 27 predictions per sex and target Olympic year.
- Predictions are ordered fastest to slowest within each output group.
- Historical Olympic evaluation uses only earlier World Championships.
- Metrics are reported separately for men, women and combined data.
- Prediction CSV files and required charts are generated.
- Training is reproducible from a saved configuration and random seed.
- Unit tests pass with `pytest`.
- A smoke-training command completes on CPU.
- The final README documents commands, file locations and known limitations.

Required shape tests:

```text
stage encoder output:       [batch, edition, 48]
edition encoder output:     [batch, edition, 64]
GRU context output:         [batch, 48]
query decoder output:       [number_of_queries, 1]
Olympic output per sex:     [27]
Olympic output both sexes:  [2, 27]
```

---

## 20. Known limitations

The total number of recorded results is much larger than the number of independent competition editions. Results within the same edition are correlated and must not be treated as independent temporal observations.

The model should therefore remain small. Regularization, early stopping, edition-level validation and uncertainty reporting are essential.

Other limitations include:

- Changes in swimsuit regulations and timing technology.
- Changes in qualification procedures.
- Missing semifinal stages in early editions.
- Irregular intervals between World Championships.
- The Tokyo 2020 edition being held in 2021.
- Olympic competitive conditions differing from World Championships.
- A future 2028 prediction remaining provisional until all pre-2028 World Championships are available.

Do not claim that a low training loss proves future Olympic accuracy.

---

## 21. Optional uncertainty estimation

After the deterministic model works correctly, add uncertainty using an ensemble of 5 independently seeded models:

```text
seeds = [11, 22, 33, 44, 55]
```

Report:

```text
ensemble mean prediction
ensemble standard deviation
empirical 95% interval
```

Do not implement uncertainty before the core data pipeline, leakage tests and deterministic model are verified.

---

## 22. Implementation order for Codex

Implement in the following order:

1. Create the project structure and configuration.
2. Load and validate both Excel workbooks.
3. Build official Olympic progression targets.
4. Build World Championships variable-length result sets.
5. Add training-only normalization.
6. Implement masked DeepSets pooling.
7. Implement edition encoder and causal GRU.
8. Implement query construction and query decoder.
9. Implement weighted Huber and ordering losses.
10. Add World Championships expanding-window training.
11. Add Olympic walk-forward evaluation.
12. Add future Olympic prediction.
13. Generate metrics, CSV files and plots.
14. Add unit tests and leakage tests.
15. Run CPU smoke training and document the result.

Do not skip validation tests in order to reach model training sooner.

---

## 23. Definition of the final model

The final system is defined as:

> A hierarchical neural forecasting model that encodes every valid men's and women's 50 m freestyle result from the heats, semifinals and finals of each World Aquatics Championships edition with stage-specific DeepSets encoders; models chronological competition development with a causal GRU; and uses a conditional Query Decoder to predict Olympic heat-to-semifinal qualifier times, semifinal-to-final qualifier times and final top-three times. Olympic results are used only for time-respecting evaluation, and no athlete identity information is used by the model.
