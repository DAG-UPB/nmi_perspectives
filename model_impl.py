## Code to run
# cd ~/scm/nmi_perspectives && nohup bash -c '
# source $(conda info --base)/etc/profile.d/conda.sh
# conda activate nmi_perspectives
# export WANDB_PROJECT="nmi_experiments"

# for seed in 410 1425 1680 1825 2287 3658 4013 4507 8936 9675; do
#     CUDA_VISIBLE_DEVICES=0 python model_impl.py --seed $seed --cuda 0 --type no_leak >> logs/log_seed_${seed}_no_leak.txt 2>&1
#     CUDA_VISIBLE_DEVICES=0 python model_impl.py --seed $seed --cuda 0 --type leak >> logs/log_seed_${seed}_leak.txt 2>&1
# done
# ' > logs/nmi_experiments.log 2>&1 &

import pandas as pd
import numpy as np
import os
from tsai.all import *
from fastai.metrics import rmse, mse
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import random
import csv
import argparse

import torch

from tsai.inference import load_learner

import seaborn as sns

from datasets import load_dataset

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

import wandb

import logging
logging.getLogger().setLevel(logging.WARNING)


def set_all_seeds(seed):
    """Set seeds for reproducibility"""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_and_preprocess(file_path):
    """Data loading and preprocessing for stock data"""
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df[["Date", "Price"]]
    df = df.set_index('Date')
    df["Price"] = df["Price"].str.replace(',', '').astype(float)
    return df


def split_df(df, domain):
    """Split dataframe into train and val (80/20)"""
    train_df = df.copy()
    val_df = df.copy()
    
    for index, row in df.iterrows():
        for col in df.columns:
            split_index = int(len(row[col]) * 0.8)
            train_df.at[index, col] = row[col][:split_index]
            val_df.at[index, col] = row[col][split_index:]
    
    train_df["domain"] = domain
    val_df["domain"] = domain
    return train_df, val_df


def scale_by_domain(train_df, val_df):
    """
    Scale values by domain using separate scalers.
    Returns scaled dataframes and dict of scalers
    """
    scalers = {}
    train_df_scaled = train_df.copy()
    val_df_scaled = val_df.copy()
    
    unique_domains = train_df['domain'].unique()
    
    for domain in unique_domains:
        scaler = MinMaxScaler()
        
        train_domain_idx = train_df['domain'] == domain
        val_domain_idx = val_df['domain'] == domain
        
        train_values = np.concatenate(train_df.loc[train_domain_idx, 'values'].values)
        scaler.fit(train_values.reshape(-1, 1))
        
        train_scaled = []
        for series in train_df.loc[train_domain_idx, 'values']:
            scaled_series = scaler.transform(series.reshape(-1, 1)).flatten()
            train_scaled.append(scaled_series)
        
        val_scaled = []
        for series in val_df.loc[val_domain_idx, 'values']:
            scaled_series = scaler.transform(series.reshape(-1, 1)).flatten()
            val_scaled.append(scaled_series)
        
        train_df_scaled.loc[train_domain_idx, 'values'] = train_scaled
        val_df_scaled.loc[val_domain_idx, 'values'] = val_scaled
        
        scalers[domain] = scaler
    
    return train_df_scaled, val_df_scaled, scalers


def create_sequences_from_multiple_series(df, seq_len, horizon):
    """Convert multiple time series into X, y sequences"""
    all_X = []
    all_y = []
    
    for series in df['values']:
        for i in range(0, len(series) - seq_len - horizon + 1, horizon):
            X_seq = series[i:(i + seq_len)]
            y_seq = series[(i + seq_len):(i + seq_len + horizon)]
            all_X.append(X_seq)
            all_y.append(y_seq)
    
    return np.array(all_X), np.array(all_y)


def run_experiment(seed, cuda_device, experiment_type, n_trials=50):
    """
    Run a single experiment (leak or no_leak).
    
    Args:
        seed: Random seed
        cuda_device: CUDA device index
        experiment_type: "leak" or "no_leak"
        n_trials: Number of Optuna trials
    """
    is_leak = experiment_type == "leak"
    exp_name = "Leak" if is_leak else "No-Leak"
    
    print(f"\n{'='*60}")
    print(f"Running {exp_name} experiment with seed: {seed}")
    print(f"{'='*60}\n")
    
    # Set seeds
    set_all_seeds(seed)
    
    # Initialize wandb
    wandb.init(
        project=os.environ.get("WANDB_PROJECT", "tst"),
        name=f"{experiment_type}_seed_{seed}",
        config={"seed": seed, "experiment_type": experiment_type, "cuda_device": cuda_device},
        reinit=True
    )
    
    # Set device
    device = torch.device(f"cuda:{cuda_device}")
    torch.cuda.set_device(device=device)
    
    # Load DAX and SP500 data
    dax_df = load_and_preprocess("data/DAX Historical Data.csv")
    sp500_df = load_and_preprocess("data/S&P 500 Historical Data.csv")
    
    # Load additional stock indices
    bovespa_df = load_and_preprocess("data/Bovespa Historical Data.csv")
    cac40_df = load_and_preprocess("data/CAC 40 Historical Data.csv")
    dowjones_df = load_and_preprocess("data/Dow Jones Industrial Average Historical Data.csv")
    ftse100_df = load_and_preprocess("data/FTSE 100 Historical Data.csv")
    nikkei225_df = load_and_preprocess("data/Nikkei 225 Historical Data.csv")
    tsx_df = load_and_preprocess("data/S&P_TSX Composite Historical Data.csv")
    
    dax_df = dax_df.sort_index()
    sp500_df = sp500_df.sort_index()
    bovespa_df = bovespa_df.sort_index()
    cac40_df = cac40_df.sort_index()
    dowjones_df = dowjones_df.sort_index()
    ftse100_df = ftse100_df.sort_index()
    nikkei225_df = nikkei225_df.sort_index()
    tsx_df = tsx_df.sort_index()
    
    # Split data based on experiment type
    if is_leak:
        # Leak
        dax_df_train = dax_df[(dax_df.index.year > 2015) & (dax_df.index < "2020-07-01")]
        dax_df_val = dax_df[(dax_df.index >= "2020-07-01") & (dax_df.index.year < 2021)]
        
        # Additional indices
        bovespa_train = bovespa_df[(bovespa_df.index.year > 2015) & (bovespa_df.index < "2020-07-01")]
        bovespa_val = bovespa_df[(bovespa_df.index >= "2020-07-01") & (bovespa_df.index.year < 2021)]
        
        cac40_train = cac40_df[(cac40_df.index.year > 2015) & (cac40_df.index < "2020-07-01")]
        cac40_val = cac40_df[(cac40_df.index >= "2020-07-01") & (cac40_df.index.year < 2021)]
        
        dowjones_train = dowjones_df[(dowjones_df.index.year > 2015) & (dowjones_df.index < "2020-07-01")]
        dowjones_val = dowjones_df[(dowjones_df.index >= "2020-07-01") & (dowjones_df.index.year < 2021)]
        
        ftse100_train = ftse100_df[(ftse100_df.index.year > 2015) & (ftse100_df.index < "2020-07-01")]
        ftse100_val = ftse100_df[(ftse100_df.index >= "2020-07-01") & (ftse100_df.index.year < 2021)]
        
        nikkei225_train = nikkei225_df[(nikkei225_df.index.year > 2015) & (nikkei225_df.index < "2020-07-01")]
        nikkei225_val = nikkei225_df[(nikkei225_df.index >= "2020-07-01") & (nikkei225_df.index.year < 2021)]
        
        tsx_train = tsx_df[(tsx_df.index.year > 2015) & (tsx_df.index < "2020-07-01")]
        tsx_val = tsx_df[(tsx_df.index >= "2020-07-01") & (tsx_df.index.year < 2021)]
    else:
        # No Leak
        dax_df_train = dax_df[dax_df.index < "2019-07-01"]
        dax_df_val = dax_df[(dax_df.index >= "2019-07-01") & (dax_df.index.year < 2020)]
        
        # Additional indices
        bovespa_train = bovespa_df[bovespa_df.index < "2019-07-01"]
        bovespa_val = bovespa_df[(bovespa_df.index >= "2019-07-01") & (bovespa_df.index.year < 2020)]
        
        cac40_train = cac40_df[cac40_df.index < "2019-07-01"]
        cac40_val = cac40_df[(cac40_df.index >= "2019-07-01") & (cac40_df.index.year < 2020)]
        
        dowjones_train = dowjones_df[dowjones_df.index < "2019-07-01"]
        dowjones_val = dowjones_df[(dowjones_df.index >= "2019-07-01") & (dowjones_df.index.year < 2020)]
        
        ftse100_train = ftse100_df[ftse100_df.index < "2019-07-01"]
        ftse100_val = ftse100_df[(ftse100_df.index >= "2019-07-01") & (ftse100_df.index.year < 2020)]
        
        nikkei225_train = nikkei225_df[nikkei225_df.index < "2019-07-01"]
        nikkei225_val = nikkei225_df[(nikkei225_df.index >= "2019-07-01") & (nikkei225_df.index.year < 2020)]
        
        tsx_train = tsx_df[tsx_df.index < "2019-07-01"]
        tsx_val = tsx_df[(tsx_df.index >= "2019-07-01") & (tsx_df.index.year < 2020)]
    
    sp500_test = sp500_df[sp500_df.index.year == 2020]
    
    # Load additional time series data
    nn5 = pd.read_parquet("data/nn5_daily.parquet")
    tourism = pd.read_parquet("data/tourism.parquet")
    
    train_nn5, val_nn5 = split_df(nn5, "NN5")
    train_tourism, val_tourism = split_df(tourism, "tourism")
    
    combined_train_df = pd.concat([train_nn5, train_tourism], ignore_index=True)
    combined_val_df = pd.concat([val_nn5, val_tourism], ignore_index=True)
    
    # Create stock index domain dataframes
    dax_domain_train_df = pd.DataFrame({"values": [dax_df_train["Price"].values], "domain": "DAX"})
    dax_domain_val_df = pd.DataFrame({"values": [dax_df_val["Price"].values], "domain": "DAX"})
    
    bovespa_domain_train_df = pd.DataFrame({"values": [bovespa_train["Price"].values], "domain": "Bovespa"})
    bovespa_domain_val_df = pd.DataFrame({"values": [bovespa_val["Price"].values], "domain": "Bovespa"})
    
    cac40_domain_train_df = pd.DataFrame({"values": [cac40_train["Price"].values], "domain": "CAC40"})
    cac40_domain_val_df = pd.DataFrame({"values": [cac40_val["Price"].values], "domain": "CAC40"})
    
    dowjones_domain_train_df = pd.DataFrame({"values": [dowjones_train["Price"].values], "domain": "DowJones"})
    dowjones_domain_val_df = pd.DataFrame({"values": [dowjones_val["Price"].values], "domain": "DowJones"})
    
    ftse100_domain_train_df = pd.DataFrame({"values": [ftse100_train["Price"].values], "domain": "FTSE100"})
    ftse100_domain_val_df = pd.DataFrame({"values": [ftse100_val["Price"].values], "domain": "FTSE100"})
    
    nikkei225_domain_train_df = pd.DataFrame({"values": [nikkei225_train["Price"].values], "domain": "Nikkei225"})
    nikkei225_domain_val_df = pd.DataFrame({"values": [nikkei225_val["Price"].values], "domain": "Nikkei225"})
    
    tsx_domain_train_df = pd.DataFrame({"values": [tsx_train["Price"].values], "domain": "TSX"})
    tsx_domain_val_df = pd.DataFrame({"values": [tsx_val["Price"].values], "domain": "TSX"})
    
    
    # Combine all training data
    combined_train_exp_df = pd.concat([
        dax_domain_train_df, 
        bovespa_domain_train_df, 
        cac40_domain_train_df,
        dowjones_domain_train_df,
        ftse100_domain_train_df,
        nikkei225_domain_train_df,
        tsx_domain_train_df,
        combined_train_df
    ], ignore_index=True)
    combined_val_exp_df = pd.concat([
        dax_domain_val_df, 
        bovespa_domain_val_df, 
        cac40_domain_val_df,
        dowjones_domain_val_df,
        ftse100_domain_val_df,
        nikkei225_domain_val_df,
        tsx_domain_val_df,
        combined_val_df
    ], ignore_index=True)
        
    # Scale data
    train_df_scaled, val_df_scaled, scalers = scale_by_domain(combined_train_exp_df, combined_val_exp_df)
    
    # Create sequences
    seq_len = 20
    horizon = 10
    
    X_train, y_train = create_sequences_from_multiple_series(train_df_scaled, seq_len, horizon)
    X_val, y_val = create_sequences_from_multiple_series(val_df_scaled, seq_len, horizon)
    
    # Create splits for tsai
    splits = (
        list(range(len(X_train) - int(0.2 * len(X_train)))),
        list(range(len(X_train) - int(0.2 * len(X_train)), len(X_train)))
    )
    
    # Prepare test data
    scaler_sp500 = MinMaxScaler()
    scaler_sp500.fit(dax_df_train["Price"].values.reshape(-1, 1))
    sp500_scaled = scaler_sp500.transform(sp500_test["Price"].values.reshape(-1, 1))
    
    X_test = []
    y_test = []
    sp500_scaled_flatten = sp500_scaled.flatten()
    
    for i in range(0, len(sp500_scaled_flatten) - seq_len - horizon + 1, horizon):
        X_seq = sp500_scaled_flatten[i:(i + seq_len)]
        y_seq = sp500_scaled_flatten[(i + seq_len):(i + seq_len + horizon)]
        X_test.append(X_seq)
        y_test.append(y_seq)
    
    X_test, y_test = np.array(X_test), np.array(y_test)
    
    print(f"\nTrain shape: {X_train.shape}, {y_train.shape}")
    print(f"Val shape: {X_val.shape}, {y_val.shape}")
    print(f"Test shape: {X_test.shape}, {y_test.shape}")
        
    # File to store the optuna results
    log_file = os.path.join("tuning", f"{experiment_type}_optuna_trials_seed_{seed}.csv")
    
    with open(log_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["trial_number", "learning_rate", "batch_size", "epochs", "dropout", "fc_dropout", 
                        "n_layers", "n_heads", "d_model", "d_ff", "val_mae"])
    
    def objective(trial):
        learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-1)
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
        epochs = trial.suggest_int("epochs", 2, 20)
        dropout = trial.suggest_uniform("dropout", 0.0, 0.7)
        fc_dropout = trial.suggest_uniform("fc_dropout", 0.0, 0.7)
        n_layers = trial.suggest_int("n_layers", 2, 8)
        n_heads = trial.suggest_categorical("n_heads", [4, 8, 16, 32])
        d_model = trial.suggest_categorical("d_model", [32, 64, 128, 256, 512])
        d_ff = trial.suggest_categorical("d_ff", [32, 64, 128, 256, 512])
        
        tfms = [None, TSForecasting()]
        
        fcst = TSForecaster(
            X_train,
            y_train,
            splits=splits,
            path='models',
            tfms=tfms,
            bs=batch_size,
            arch="TSTPlus",
            arch_config={
                "dropout": dropout,
                "n_layers": n_layers,
                "d_model": d_model,
                "n_heads": n_heads,
                "fc_dropout": fc_dropout,
                "d_ff": d_ff
            },
            metrics=mae,
        )
        
        fcst.fit_one_cycle(epochs, learning_rate)
        
        X_val_reshaped = X_val.reshape(X_val.shape[0], 1, X_val.shape[1])
        raw_preds, target, _ = fcst.get_X_preds(X_val_reshaped, y_val)
        val_mae = mae(target, raw_preds)
        
        with open(log_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([trial.number, learning_rate, batch_size, epochs, dropout, fc_dropout, 
                           n_layers, n_heads, d_model, d_ff, val_mae])
        
        return val_mae
    
    # Run the Optuna study
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials)
    
    print(f"\nBest hyperparameters: {study.best_params}")
    
    config = study.best_params
    
    # Combine train and val for final model training
    X_final = np.concatenate([X_train, X_val], axis=0)
    y_final = np.concatenate([y_train, y_val], axis=0)
    
    n_train_final = int(0.8 * len(X_final))
    final_splits = (
        list(range(n_train_final)),
        list(range(n_train_final, len(X_final)))
    )
    
    print(f"\nTraining final model on combined data: {X_final.shape}")
    
    # Train final model with best hyperparameters
    tfms = [None, TSForecasting()]
    
    fcst = TSForecaster(
        X_final,
        y_final,
        splits=final_splits,
        path='models',
        tfms=tfms,
        bs=config["batch_size"],
        arch="TSTPlus",
        arch_config={
            "dropout": config["dropout"],
            "fc_dropout": config["fc_dropout"],
            "n_layers": config["n_layers"],
            "n_heads": config["n_heads"],
            "d_model": config["d_model"],
            "d_ff": config["d_ff"]
        },
        metrics=mae
    )
    
    fcst.fit_one_cycle(config["epochs"], config["learning_rate"])
    
    # Export model
    model_filename = f"fcst_{experiment_type}_seed_{seed}.pkl"
    fcst.export(model_filename)
    print(f"\nModel saved as: {model_filename}")
    
    # Evaluate on test set
    X_test_reshaped = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
    raw_preds, target, preds = fcst.get_X_preds(X_test_reshaped, y_test)
    test_mae = float(mae(target, raw_preds).item())
    test_rmse = float(rmse(target, raw_preds).item())
    
    print(f"\nTest Results (SP500 2020):")
    print(f"  MAE: {test_mae:.6f}")
    print(f"  RMSE: {test_rmse:.6f}")
    
    # Log to wandb
    wandb.log({
        "test_mae": test_mae,
        "test_rmse": test_rmse,
        **config
    })
    wandb.finish()
    
    # Save test results
    results_file = os.path.join("tuning", f"results_{experiment_type}_seed_{seed}.csv")
    with open(results_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["seed", "test_mae", "test_rmse", "best_params"])
        writer.writerow([seed, test_mae, test_rmse, str(config)])
    
    print(f"Results saved as: {results_file}")
    
    return test_mae, test_rmse, config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run model training with specified seed and experiment type")
    parser.add_argument("--seed", type=int, required=True, help="Random seed for reproducibility")
    parser.add_argument("--cuda", type=int, default=0, help="CUDA device to use")
    parser.add_argument("--n_trials", type=int, default=50, help="Number of Optuna trials")
    parser.add_argument("--type", type=str, required=True, choices=["leak", "no_leak"], 
                        help="Experiment type: 'leak' or 'no_leak'")
    
    args = parser.parse_args()
    
    run_experiment(args.seed, args.cuda, args.type, args.n_trials)
