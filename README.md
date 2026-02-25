# NMI Perspectives — Data Leakage in Time Series Forecasting

This repository contains a study on data leakage in time series foundation model evaluations. It demonstrates how temporal overlap between training and test data can inflate performance metrics, using stock index forecasting as a case study.

## Overview

A Time Series Transformer model is trained on multiple time series domains and evaluated on S&P 500 data for the year 2020. Two experimental conditions are compared across 10 random seeds:

- **Leak**: Training data from auxiliary stock indices (Bovespa, DAX, Dow Jones, CAC40, FTSE 100, Nikkei 225, TSX) overlaps temporally with the S&P 500 test period (2016–mid 2020 for train, mid 2020– end 2020 for validation).
- **No-Leak**: Auxiliary stock index training data is strictly before the test period (pre-July 2019 for train, July 2019– end 2020 for validation).

Both setups also include NN5 Daily and Tourism datasets as additional training domains.

## Repository Structure

| Path | Description |
|---|---|
| `model_impl.py` | Main experiment script — data loading, Optuna hyperparameter tuning, and model training |
| `seeds.ipynb` | Generates the 10 random seeds used across experiments |
| `test_leakage.ipynb` | Analysis notebook — loads results, computes statistics, and generates comparison plots |
| `data/` | Stock index CSVs, plus `nn5_daily.parquet` and `tourism.parquet` |
| `models/` | Exported trained model checkpoints (`.pkl`) for each seed × condition |
| `tuning/` | Optuna trial logs and final test results (MAE, RMSE) per seed × condition |
| `logs/` | Training stdout logs per seed |
| `plots/` | Generated forecast plots (per-seed and final combined plot) |
| `lineage-analysis/` | CSV mapping time series foundation models to their training/evaluation datasets |
| `requirements.txt` | Full pip dependency list |

## Data

The stock index data was downloaded from [investing.com](https://www.investing.com/) (date range 01/02/2015–01/01/2021) and needs to be placed into the `data/` folder to reproduce our experiments and results.

## How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Run a single experiment (leak or no_leak)
python model_impl.py --seed 410 --cuda 0 --type leak --n_trials 50
python model_impl.py --seed 410 --cuda 0 --type no_leak --n_trials 50

# Run all 10 seeds (both conditions)
for seed in 410 1425 1680 1825 2287 3658 4013 4507 8936 9675; do
    python model_impl.py --seed $seed --cuda 0 --type no_leak
    python model_impl.py --seed $seed --cuda 0 --type leak
done
```

## Seeds

10 seeds generated deterministically (`random.seed(42)`): 410, 1425, 1680, 1825, 2287, 3658, 4013, 4507, 8936, 9675

## Lineage Analysis

`lineage-analysis/Time-Series-Foundation-Models-Lineage.csv` maps 30+ time series foundation models (Chronos, TimesFM, Moirai, etc.) to the datasets used in their pre-training (P), zero-shot evaluation (ZS), or transfer/test (T/T) across domains and frequencies.
