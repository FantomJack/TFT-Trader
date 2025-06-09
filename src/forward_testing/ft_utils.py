import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


# ============================
# 1. Checkpoints Loading Function
# ============================
def load_checkpoint_paths(root_folder: str, folders_to_scan: str):
    """
    Scans specified subfolders in a root directory to find all model checkpoint files.

    Args:
        root_folder (str): Path to the root directory containing subfolders of experiments.
        folders_to_scan (list of str): List of subfolder names (hyperparameter configs) to scan.

    Returns:
        list of dict: Each dict contains details about a checkpoint file:
            - 'path': Full path to the checkpoint file.
            - 'hyperparams': Hyperparameter setting (subfolder).
            - 'data_split': Data split identifier.
            - 'training_split': Training split identifier.
            - 'filename': Name of the checkpoint file.
            - 'idx': Index of the checkpoint file in its folder.
    """
    checkpoint_paths = []

    for hyperparams in folders_to_scan:
        checkpoints_folder = os.path.join(root_folder, hyperparams)
        if os.path.isdir(checkpoints_folder):
            for data_split in os.listdir(checkpoints_folder):
                data_split_path = os.path.join(checkpoints_folder, data_split)
                if os.path.isdir(data_split_path):
                    for training_split in os.listdir(data_split_path):
                        idx = 0
                        training_subfolder_path = os.path.join(data_split_path, training_split)
                        if os.path.isdir(training_subfolder_path):
                            for filename in os.listdir(training_subfolder_path):
                                if filename.endswith('.ckpt') or filename.endswith('.keras'):
                                    idx += 1
                                    full_path = os.path.join(training_subfolder_path, filename)
                                    checkpoint_paths.append({
                                        'path': full_path,
                                        'hyperparams': hyperparams,
                                        'data_split': data_split,
                                        'training_split': training_split,
                                        'filename': filename,
                                        'idx': idx
                                    })
    return checkpoint_paths


# ============================
# 2. Evaluation Function
# ============================
def compute_metrics(df: pd.DataFrame) -> (pd.DataFrame, dict):
    """
    Calculates key performance metrics for a trading strategy based on predictions.

    Adds columns for strategy and market returns, and computes cumulative returns,
    Sharpe ratio, and maximum drawdown.

    Args:
        df (pd.DataFrame): DataFrame containing at least 'signal', 'actual_return', and 'Close' columns.

    Returns:
        (pd.DataFrame, dict): Tuple containing:
            - Updated DataFrame with additional columns.
            - Dictionary of computed metrics:
                * 'total_return': Final strategy return.
                * 'sharpe_ratio': Annualized Sharpe ratio.
                * 'max_drawdown': Maximum drawdown.
                * 'drawdown_series': Series of drawdown values (for plotting).
    """
    cost_per_trade = 0.0005  # 0.05% per trade

    # Calculate strategy returns and cumulative returns
    df['strategy_return'] = df['signal'] * df['actual_return'] - cost_per_trade * df['signal'].abs()
    df['cumulative_strategy'] = (1 + df['strategy_return']).cumprod()

    # Calculate market returns and cumulative returns
    df['market_return'] = df['Close'].pct_change().fillna(0)
    df['cumulative_market'] = (1 + df['market_return']).cumprod()

    # Calculate performance metrics
    total_return = df['cumulative_strategy'].iloc[-1] - 1
    avg_daily = df['strategy_return'].mean()
    std_daily = df['strategy_return'].std()
    sharpe_ratio = (avg_daily / std_daily * np.sqrt(252)) if std_daily != 0 else np.nan
    running_max = df['cumulative_strategy'].cummax()
    drawdown = (running_max - df['cumulative_strategy']) / running_max
    max_drawdown = drawdown.max()

    metrics = {
        "total_return": total_return,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
        "drawdown_series": drawdown  # for plotting purposes
    }
    return df, metrics


# ============================
# 3. Forward testing Function
# ============================
def forwardtest_strategy(data: pd.DataFrame, predictions, ticker: str, plot: bool = False):
    """
    Forward tests a basic long-only strategy using predicted daily return.

    Parameters:
        data (pd.DataFrame): DataFrame with at least the following columns:
            - Date: The trading date.
            - Open: Daily open price.
            - Close: Daily close price.
            - volatility: Daily volatility.
        predictions (array-like): Predicted daily returns (in percentage).
        ticker (str): Ticker name, used for plot file naming.
        plot (bool): If True, plots results.

    Returns:
        eval_data (pd.DataFrame): DataFrame with daily forward test results.
        metrics (dict): Computed performance metrics.
    """

    eval_data = pd.DataFrame()
    eval_data[['Date', 'Close', 'volatility', 'growth']] = data[['Date', 'Close', 'volatility', 'growth']]

    data = data.sort_values('Date').reset_index(drop=True)
    if len(data) != len(predictions):
        raise ValueError("Length of data and predictions do not match.")

    eval_data['actual_return'] = eval_data['growth'] / 100
    eval_data['predicted_return'] = np.array(predictions) / 100

    # eval_data['signal'] = (predictions > 0.0).astype(int)
    # eval_data['signal'] = np.where(predictions > 0.0, 1,
    #                         np.where(predictions < 0.0, -1, 0))

    conditions = [
        predictions > 1,
        (predictions > 0) & (predictions <= 1),
        # (predictions >= -0.5) & (predictions <= 0.5),
        (predictions < 0) & (predictions >= -1),
        predictions < -1
    ]
    # choices = [1, 0.5, 0, -0.5, -1]
    choices = [1, 0.5, -0.5, -1]
    eval_data['signal'] = np.select(conditions, choices, default=0)

    # eval_data['signal'] = np.clip(predictions/10, -1, 1)


    eval_data, metrics = compute_metrics(eval_data)
    if plot:
        plot_results(eval_data, metrics, ticker)

    print("Total Strategy Return: {:.2%}".format(metrics['total_return']))
    print("Annualized Sharpe Ratio: {:.2f}".format(metrics['sharpe_ratio']))
    print("Maximum Drawdown: {:.2%}".format(metrics['max_drawdown']))

    return eval_data, metrics


# ==========================================
# 4. Aggregated Results Computing Function
# ==========================================
def aggregate_results(results_all: dict) -> (pd.DataFrame, dict):
    """
    Aggregates forward test results across multiple tickers and computes overall metrics.

    Args:
        results_all (dict): Mapping {ticker: DataFrame}, where each DataFrame includes
                            'Date', 'strategy_return', and 'market_return'.

    Assumes each ticker DataFrame has a 'Date', 'strategy_return', and 'market_return' column.

    Returns:
        aggregated (pd.DataFrame): DataFrame containing:
            - 'aggregated_daily_return': mean of daily strategy returns across tickers
            - 'aggregated_cumulative': cumulative aggregated strategy return
            - 'aggregated_market_daily_return': mean of daily market returns across tickers
            - 'aggregated_market_cumulative': cumulative aggregated market return
        aggregated_metrics (dict): Contains:
            - 'total_return': final aggregated cumulative return minus 1
            - 'sharpe_ratio': annualized Sharpe ratio (using 252 trading days)
            - 'max_drawdown': maximum drawdown of the aggregated strategy
    """
    # Create dictionaries for individual strategy and market returns
    strat_returns = {}
    market_returns = {}

    for ticker, df in results_all.items():
        df = df.copy()
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        strat_returns[ticker] = df['strategy_return']
        market_returns[ticker] = df['market_return']

    # Combine individual series into DataFrames
    combined_strat = pd.DataFrame(strat_returns)
    combined_market = pd.DataFrame(market_returns)

    # Compute aggregated daily returns (mean across tickers)
    combined_strat['aggregated_daily_return'] = combined_strat.mean(axis=1)
    combined_market['aggregated_market_daily_return'] = combined_market.mean(axis=1)

    # Compute cumulative returns
    combined_strat['aggregated_cumulative'] = (1 + combined_strat['aggregated_daily_return']).cumprod()
    combined_market['aggregated_market_cumulative'] = (1 + combined_market['aggregated_market_daily_return']).cumprod()

    # Merge into one aggregated DataFrame
    aggregated = pd.concat([combined_strat[['aggregated_daily_return', 'aggregated_cumulative']],
                            combined_market[['aggregated_market_daily_return', 'aggregated_market_cumulative']]],
                           axis=1)
    aggregated.reset_index(inplace=True)

    # Calculate performance metrics for the aggregated strategy
    avg_daily = aggregated['aggregated_daily_return'].mean()
    std_daily = aggregated['aggregated_daily_return'].std()
    sharpe_ratio = (avg_daily / std_daily * np.sqrt(252)) if std_daily != 0 else np.nan

    running_max = aggregated['aggregated_cumulative'].cummax()
    drawdown = (running_max - aggregated['aggregated_cumulative']) / running_max
    max_drawdown = drawdown.max()

    total_return = aggregated['aggregated_cumulative'].iloc[-1] - 1

    aggregated_metrics = {
        "total_return": total_return,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
    }

    return aggregated, aggregated_metrics


def print_signals(result):
    # Set display options to show the full DataFrame
    pd.set_option('display.max_rows', None)  # Show all rows
    pd.set_option('display.max_columns', None)  # Show all columns
    pd.set_option('display.width', None)  # Adjust width to screen
    pd.set_option('display.max_colwidth', None)  # Show full content in each column

    print(result)

    pd.reset_option('display.max_rows')
    pd.reset_option('display.max_columns')
    pd.reset_option('display.width')
    pd.reset_option('display.max_colwidth')


def plot_results(df: pd.DataFrame, metrics: dict, filename="ticker"):
    """
    Plots the equity curve and drawdown for a strategy vs. the market.

    Args:
        df (pd.DataFrame): DataFrame containing columns 'Date', 'cumulative_strategy', 'cumulative_market'.
        metrics (dict): Dictionary containing a 'drawdown_series' key.
        filename (str): Name for saving the output plot (without extension).
    """

    fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(14, 9), sharex=True,
                                        gridspec_kw={'height_ratios': [3, 1]})

    # Equity curve plot: Strategy vs. Market
    ax1.plot(df['Date'], df['cumulative_strategy'], label='stratégia', color='blue')
    ax1.plot(df['Date'], df['cumulative_market'], label='trh', linestyle='--', color='orange')
    ax1.set_title('Krivka návratnosti kapitálu: stratégia vs. trh')
    ax1.set_ylabel('Kumulatívny výnos')
    ax1.legend()
    ax1.grid(True)

    # # Volatility plot
    # ax2.plot(df['Date'], df['volatility'], label='Volatility', color='green')
    # ax2.set_title('Volatility')
    # ax2.set_ylabel('Volatility')
    # ax2.legend()
    # ax2.grid(True)

    # Drawdown plot: Strategy only
    ax3.fill_between(df['Date'], metrics['drawdown_series'], color='red', alpha=0.4)
    ax3.set_title('Drawdown stratégie')
    ax3.set_ylabel('drawdown')
    ax3.set_xlabel('dátum')
    ax3.grid(True)


    plt.tight_layout()
    plt.savefig('../../forward_testing/plots/' + filename + ".png")  # Save the figure


def plot_aggregated_results(aggregated: pd.DataFrame, filename: str = "aggregated_results.png"):
    """
    Plots aggregated strategy and market cumulative returns.

    Args:
        aggregated (pd.DataFrame): DataFrame with 'Date', 'aggregated_cumulative', 'aggregated_market_cumulative'.
        filename (str): File path to save the plot.
    """


    fig, ax = plt.subplots(figsize=(14, 8))

    ax.plot(aggregated['Date'], aggregated['aggregated_cumulative'], label='agregovaná stratégia', color='blue')
    ax.plot(aggregated['Date'], aggregated['aggregated_market_cumulative'], label='agregovaný trh', linestyle='--',
            color='orange')

    ax.set_title("Agregovaná krivka hodnoty kapitálu: stratégia vs. trh")
    ax.set_xlabel("dátum")
    ax.set_ylabel('Kumulatívny výnos')
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    plt.savefig(f"../../forward_testing/{filename}.png")