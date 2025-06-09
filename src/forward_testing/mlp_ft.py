import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ft_utils import (compute_metrics, plot_results, forwardtest_strategy, print_signals,
                      aggregate_results, plot_aggregated_results, load_checkpoint_paths, )

class AI:
    MATRIX_ROWS_SIZE = 30
    X_COLUMNS = ['growth', 'Volume', 'sma_5', 'volatility', 'rsi', 'atr_rel', 'cmf', 'month', 'day_of_week']
    ADAM_LEARNING_RATE = 0.001


def create_windows(data, window_size):
    windows = []
    for i in range(len(data) - window_size):
        window = data.iloc[i:(i + window_size)][AI.X_COLUMNS].values
        windows.append(window)
    return np.array(windows)


def df_prep(tickers):
    test_folder = '../../csv_data/test/'
    X_list = []
    y_list = []

    for label in tickers:
        df = pd.read_csv(f"{test_folder}{label}.csv", index_col=0, parse_dates=True)
        df.sort_index(inplace=True)
        df = df.reset_index().rename(columns={'index': 'Date'})
        df = df[-217:]

        X_list.append(create_windows(df, AI.MATRIX_ROWS_SIZE))

        y = df[AI.MATRIX_ROWS_SIZE:]
        y_list.append(y)

    X = np.concatenate(X_list, axis=0)
    y = pd.concat(y_list, axis=0)

    return X, y

if __name__ == "__main__":
    tickers = [
        'AAPL', 'ACN', 'ADBE', 'ADI', 'ADP', 'ALTR', 'AMAT', 'AMD', 'AMZN', 'AVGO', 'BA', 'BIDU', 'DE',
        'DELL', 'DIS', 'EA', 'EBAY', 'GOOG', 'GOOGL', 'HPE', 'HPQ', 'IBM', 'INTC', 'MSFT', 'NVDA'
    ]

    root_folder = '../../checkpoints/MLP/'
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

        model = tf.keras.models.load_model(
            cp,
            custom_objects={'Huber': tf.keras.losses.Huber}
        )

        print(f"Evaluating model checkpoint: {cp}")
        results_all = {}
        for ticker in tickers:
            print(f"Forward testing for {ticker}")
            X_test, y_test = df_prep([ticker])

            predictions = model.predict(X_test, batch_size=128).flatten()
            results_df, metrics = forwardtest_strategy(y_test, predictions, ticker)
            results_all[ticker] = results_df
            print("\n")

            ticker_records.append({
                "model": "MLP",  # "TFT" or "MLP"
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
            "model": "MLP",  # "TFT" or "MLP"
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
    df.to_csv("../../forward_testing/mlp_metrics.csv", index=False)
    df2 = pd.DataFrame(ticker_records)
    df2.to_csv("../../forward_testing/mlp_metrics_ticker.csv", index=False)