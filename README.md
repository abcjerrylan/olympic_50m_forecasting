# Olympic 50 m Freestyle Forecasting

First working implementation of the DeepSets + causal GRU + Query Decoder
specification in `README_Olympic_50m_Freestyle_DeepSets_GRU.md`.

中文模型规范见 `README_Olympic_50m_Freestyle_DeepSets_GRU_zh-CN.md`。

## Data

The default raw files are:

```text
data/World_Aquatics_Championships_50m_Freestyle_Complete_Results_1986-2025.xlsx
data/Olympic_Games_50m_Freestyle_Complete_Results_1988-2024.xlsx
```

The workbooks are read-only inputs. Olympic results are used only to build
evaluation labels. Names are temporary join keys and never model features.

## Commands

Run commands from this directory.

```powershell
python scripts/prepare_data.py
python scripts/train.py --config configs/base.yaml --output-dir outputs/runs/main
python scripts/evaluate_olympics.py --config configs/base.yaml --years 2000 2004 2008 2012 2016 2020 2024 --output-dir outputs/runs/walk_forward
python scripts/predict_olympics.py --checkpoint outputs/runs/main/best_model.pt --target-year 2028 --output outputs/runs/main/olympics_2028_predictions.csv
pytest -q
```

For a quick CPU check:

```powershell
python scripts/train.py --config configs/base.yaml --max-epochs 3 --patience 2 --output-dir outputs/runs/smoke
```

## Leakage controls

- Context construction rejects World Championships editions at or after the
  target year.
- Scalers are fitted once on the training years supplied by the caller.
- Olympic rows never enter the training-example builder.
- Each loss step contains all queries from one target edition and sex.
- Model tensors contain only numeric result, phase, year, sex and count data.
- Empty phases remain empty and are represented by presence masks.

## Known limitations

The number of independent editions is small. This implementation therefore
uses a compact model, edition-level early stopping and time-respecting folds.
The 2024 Olympics should remain locked during model selection. A 2028 forecast
is provisional until all World Championships before 2028 are available.
