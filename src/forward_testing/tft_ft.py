import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import lightning.pytorch as pl
from pytorch_forecasting import TimeSeriesDataSet, GroupNormalizer, TemporalFusionTransformer, QuantileLoss
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from ft_utils import (compute_metrics, plot_results, forwardtest_strategy, print_signals,
                      aggregate_results, plot_aggregated_results, load_checkpoint_paths, )


# ============================
# 1. Data Preparation Function
# (Adapted from model_4_correct.py)
# ============================
def df_prep_TFT(tickers):
    test_folder = '../../csv_data/test/'
    df_list = []

    for label in tickers:
        tmp = pd.read_csv(test_folder + label + '.csv', parse_dates=["Date"])
        tmp.sort_values("Date", inplace=True)
        tmp['ticker'] = label
        tmp.reset_index(drop=True, inplace=True)
        tmp['idx'] = range(len(tmp))
        df_list.append(tmp)

    df = pd.concat(df_list).reset_index(drop=True)
    df = df[-217:]
    return df


# ============================
# 4. Prediction Generation using TFT
# (Adapted from model_4_correct.py)
# ============================
def generate_predictions(data, checkpoint_path, target='growth', covariates=['growth', 'Volume', 'sma_5', 'volatility'], known=[]):
    # Define forecasting parameters
    max_prediction_length = 3
    max_encoder_length = 30

    prediction_dataset = TimeSeriesDataSet(
        data,
        time_idx="idx",
        target=target,
        group_ids=['ticker'],
        static_categoricals=["ticker"],
        min_encoder_length=max_encoder_length,  # we require a full encoder window
        max_encoder_length=max_encoder_length,
        min_prediction_length=1,
        max_prediction_length=max_prediction_length,
        time_varying_known_reals=known,  # adjust if you have any known inputs
        time_varying_unknown_reals=covariates,
        target_normalizer=GroupNormalizer(method="standard", groups=["ticker"]),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True
    )

    prediction_dataloader = prediction_dataset.to_dataloader(train=False, batch_size=128, num_workers=0)

    try:
        model = TemporalFusionTransformer.load_from_checkpoint(checkpoint_path)
        print(f"Loaded model from {checkpoint_path}")
    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint from {checkpoint_path}. Error: {e}")

    trainer_cfg = dict(
        logger=False,  # no logs at all
        enable_checkpointing=False,  # no checkpoints
        enable_progress_bar=False,  # no progress bar
    )
    raw_predictions = model.predict(prediction_dataloader, mode="raw", return_x=True, num_workers=0,
                                    trainer_kwargs=trainer_cfg,)
    predictions = raw_predictions.output.prediction.cpu().numpy()

    return predictions, max_encoder_length


# ============================
# 5. Main Integration
# ============================
if __name__ == '__main__':
    tickers = ['AAPL', 'ACN', 'ADBE', 'ADI', 'ADP', 'ALTR', 'AMAT', 'AMD', 'AMZN', 'AVGO', 'BA', 'BIDU', 'DE',
                     'DELL', 'DIS', 'EA', 'EBAY', 'GOOG', 'GOOGL', 'HPE', 'HPQ', 'IBM', 'INTC', 'MSFT', 'NVDA']

    covariates = ['growth', 'Volume', 'sma_5', 'volatility', 'rsi', 'atr_rel', 'cmf']
    known_reals = ['month', 'day_of_week']

    root_folder = '../../checkpoints/TFT/'
    folders_to_scan = ['01', '02', '03']

    checkpoint_paths = load_checkpoint_paths(root_folder, folders_to_scan)
    checkpoint_metrics = {}
    records = []
    ticker_records = []

    for cp_info in checkpoint_paths:
        cp = cp_info['path']
        hyperparams = cp_info['hyperparams']
        training_split = cp_info['training_split']
        data_split = cp_info['data_split']
        filename = cp_info['filename']
        idx = cp_info['idx']

        print(f"Evaluating model checkpoint: {cp}")
        results_all = {}
        for ticker in tickers:
            print(f"Forwardtesting for {ticker}")
            data_ticker = df_prep_TFT([ticker])

            predictions, max_encoder_length = generate_predictions(
                data_ticker, cp, target='growth',
                covariates=covariates, known=known_reals
            )

            price_data = data_ticker[['Date', 'Open', 'Close', 'growth', 'volatility']].iloc[
                         max_encoder_length:max_encoder_length + len(predictions)
                         ].reset_index(drop=True)

            if len(price_data) != len(predictions):
                min_len = min(len(price_data), len(predictions))
                price_data = price_data.iloc[:min_len]
                predictions = predictions[:min_len]

            results_df, metrics = forwardtest_strategy(price_data, predictions[:, 0, 2], ticker)
            results_all[ticker] = results_df
            print("\n")

            ticker_records.append({
                "model": "TFT",  # "TFT" or "MLP"
                "hyperparams": hyperparams,
                "start_date": data_split,
                "train_val_ratio": training_split,
                "run_id": idx,
                "filename": filename,
                'ticker': ticker,
                "total_return": metrics["total_return"],
                "sharpe_ratio": metrics["sharpe_ratio"],
                "max_drawdown": metrics["max_drawdown"],
            })

        aggregated, aggregated_metrics = aggregate_results(results_all)

        records.append({
            "model": "TFT",  # "TFT" or "MLP"
            "hyperparams": hyperparams,
            "start_date": data_split,
            "train_val_ratio": training_split,
            "run_id": idx,
            "filename": filename,
            "total_return": aggregated_metrics["total_return"],
            "sharpe_ratio": aggregated_metrics["sharpe_ratio"],
            "max_drawdown": aggregated_metrics["max_drawdown"],
        })
        checkpoint_metrics[cp] = aggregated_metrics


    print("Comparison of all model checkpoints:")
    for cp, metrics in checkpoint_metrics.items():
        print(f"{cp}: Total Return={metrics['total_return']:.2%}, "
              f"Sharpe Ratio={metrics['sharpe_ratio']:.2f}, "
              f"Max Drawdown={metrics['max_drawdown']:.2%}")

    df = pd.DataFrame(records)
    df.to_csv("../../forward_testing/tft_metrics.csv", index=False)
    df2 = pd.DataFrame(ticker_records)
    df2.to_csv("../../forward_testing/tft_metrics_ticker.csv", index=False)