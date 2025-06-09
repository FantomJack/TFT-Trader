import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ft_utils import (compute_metrics, plot_results, forwardtest_strategy, print_signals,
                      aggregate_results, plot_aggregated_results, load_checkpoint_paths, )

class AI:
    MATRIX_ROWS_SIZE = 30
    X_COLUMNS = ['growth', 'Volume', 'sma_5', 'volatility', 'rsi', 'atr_rel', 'cmf']
    ADAM_LEARNING_RATE = 0.001


def create_windows(data, window_size):
    windows = []
    for i in range(len(data) - window_size):
        window = data.iloc[i:(i + window_size)][AI.X_COLUMNS].values
        windows.append(window)
    return np.array(windows)


def df_prep(tickers):
    test_folder = '/home/xbelan/Bakalarka/csv_data/daily/return_split/test/'
    X_list = []
    y_list = []

    for label in tickers:
        print(f"Forward testing for {label}")
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
    tickers = ['AAPL', 'ACN', 'ADBE', 'ADI', 'ADP', 'ALTR', 'AMAT', 'AMD', 'AMZN', 'AVGO', 'BA', 'BIDU', 'DE',
                     'DELL', 'DIS', 'EA', 'EBAY', 'GOOG', 'GOOGL', 'HPE', 'HPQ', 'IBM', 'INTC', 'MSFT', 'NVDA']
    # tickers = ['AAPL']

    results_all = {}
    model = tf.keras.models.load_model(
        'best_model-1.keras',
        custom_objects={'Huber': tf.keras.losses.Huber}
    )

    for ticker in tickers:
        print(f"Forward testing for {ticker}")
        X_test, y_test = df_prep([ticker])

        predictions = model.predict(X_test, batch_size=128).flatten()
        # mse = ((y_test['growth'] - predictions) ** 2).mean()
        results_df, metrics = forwardtest_strategy(y_test, predictions, ticker, True)
        results_all[ticker] = results_df
        print("\n")

    aggregated, aggregated_metrics = aggregate_results(results_all)

    plot_filename = f"best_model.png"
    print(f"Plotting {plot_filename}")
    plot_aggregated_results(aggregated, filename=plot_filename)

    print("Total Strategy Return: {:.2%}".format(aggregated_metrics["total_return"]))
    print("Annualized Sharpe Ratio: {:.2f}".format(aggregated_metrics["sharpe_ratio"]))
    print("Maximum Drawdown: {:.2%}".format(aggregated_metrics["max_drawdown"]))
    print("\n")