import pandas as pd
from datetime import datetime
from tft_wrapper import TFTModelWrapper, df_prep

folder = '../csv_data/'
train_tickers = ['AAPL', 'ACN', 'ADBE', 'ADI', 'ADP', 'ALTR', 'AMAT', 'AMD', 'AMZN', 'AVGO', 'BA', 'BIDU', 'DE', 'DELL',
                 'DIS', 'EA', 'EBAY', 'GOOG', 'GOOGL', 'HPE', 'HPQ', 'IBM', 'INTC', 'MSFT', 'NVDA']

cutoffs = [{'num': 0.85, 'str': '85/'}, {'num': 0.90, 'str': '90/'}, {'num': 0.95, 'str': '95/'}]
model_dimensions = [
    {'size': 32, 'lstm': 1, 'heads': 2, 'folder': '01/'},
    {'size': 64, 'lstm': 2, 'heads': 4, 'folder': '02/'},
    {'size': 128, 'lstm': 4, 'heads': 8, 'folder': '03/'},
]

for model_d in model_dimensions:

    # INITIALIZATION
    covariates = ['growth', 'Volume', 'sma_5', 'volatility', 'rsi', 'atr_rel', 'cmf']
    known_reals = ['month', 'day_of_week']
    hyperparams = {
        "batch_size": 128,
        "max_encoder_length": 30,
        "max_prediction_length": 3,
        "lstm_layers": model_d['lstm'],
        "learning_rate": 0.005,
        "hidden_size": model_d['size'],
        "attention_head_size": model_d['heads'],
        "dropout": 0.3,
        "hidden_continuous_size": model_d['size'],
        "log_interval": 10,
        "reduce_on_plateau_patience": 4,
        "quantiles": [0.1, 0.25, 0.5, 0.75, 0.9]
    }

    model_wrapper = TFTModelWrapper('growth', covariates, known_reals, hyperparams, 8)

    for dataset in range(1,5):
        data = df_prep(train_tickers, folder + str(dataset) + '/')
        for cutoff in cutoffs:
            print('../checkpoints/new/' + model_d['folder'] + str(dataset) + '/' + cutoff["str"])
            training_cutoff = int(data['idx'].max() * cutoff['num'])
            train_data = data[lambda x: x.idx <= training_cutoff]
            val_data = data[lambda x: x.idx > training_cutoff]

            for idx in range(20):
                model_wrapper.prepare_datasets(train_data, val_data)
                model_wrapper.train(1000, 10,
                                    '../checkpoints/TFT/new/' + model_d['folder'] + str(dataset) + '/' + cutoff["str"],
                                    1, True)