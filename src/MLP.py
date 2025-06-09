import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def df_prep(tickers, folder):
    X_list = []
    y_list = []

    for label in tickers:
        df = pd.read_csv(f"{folder}{label}.csv", index_col=0, parse_dates=True)
        df.sort_index(inplace=True)

        # windows: shape (n_windows, window_size, n_features)
        windows = create_windows(df, AI.MATRIX_ROWS_SIZE)
        X_list.append(windows)

        # targets: the “growth” value immediately _after_ each window
        y = df['growth'].values[AI.MATRIX_ROWS_SIZE:]
        y_list.append(y)

    # Stack everything into one big array
    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)

    return X, y

class AI:
    MATRIX_ROWS_SIZE = 30
    X_COLUMNS = ['growth', 'Volume', 'sma_5', 'volatility', 'rsi', 'atr_rel', 'cmf', 'month', 'day_of_week']
    ADAM_LEARNING_RATE = 0.001


def create_model_mlp(input_shape=(AI.MATRIX_ROWS_SIZE, len(AI.X_COLUMNS)),
                      fc_dim=64,
                      dropout=0.05,
                      activation='tanh'):

    inputs = tf.keras.Input(shape=input_shape, dtype=tf.float32)

    print('Input shape and dtype: ', inputs.shape, inputs.dtype)

    x = tf.keras.layers.BatchNormalization(axis=-1)(inputs)

    x = tf.keras.layers.Flatten()(x)

    x = tf.keras.layers.Dense(fc_dim, activation=activation)(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Dense(fc_dim//2, activation=activation)(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Dense(fc_dim//4, activation=activation)(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    outputs = tf.keras.layers.Dense(1)(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    opt = tf.keras.optimizers.Adam(learning_rate=2 * AI.ADAM_LEARNING_RATE, clipnorm=1.0)

    model.compile(optimizer=opt, loss=tf.keras.losses.Huber(), metrics=['mse'])

    model.summary()

    return model


def create_windows(data, window_size):
    windows = []
    for i in range(len(data) - window_size):
        window = data.iloc[i:(i + window_size)][AI.X_COLUMNS].values
        windows.append(window)
    return np.array(windows)


if __name__ == "__main__":

    tickers = ['AAPL', 'ACN', 'ADBE', 'ADI', 'ADP', 'ALTR', 'AMAT', 'AMD', 'AMZN', 'AVGO', 'BA', 'BIDU', 'DE',
                     'DELL', 'DIS', 'EA', 'EBAY', 'GOOG', 'GOOGL', 'HPE', 'HPQ', 'IBM', 'INTC', 'MSFT', 'NVDA']
    cutoffs = [{'num': 0.15, 'str': '85/'}, {'num': 0.1, 'str': '90/'}, {'num': 0.05, 'str': '95/'}]
    model_dimensions = [{'folder': '01/', 'n' : 32}, {'folder': '02/', 'n' : 64}, {'folder': '03/', 'n' : 128}]

    for dim in model_dimensions:
        for dataset in range(1, 5):
            folder = f"../csv_data/{dataset}/"
            X, y = df_prep(tickers, folder)

            for cutoff in cutoffs:
                X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=cutoff["num"], random_state=42)
                for run_idx in range(1, 2):
                    model = create_model_mlp(fc_dim=dim["n"])
                    filepath = f"../checkpoints/MLP/new/{dim['folder']}{dataset}/{cutoff['str']}best_model-{run_idx}.keras"

                    checkpoint_cb = ModelCheckpoint(
                        filepath=filepath,
                        monitor="val_loss",
                        save_best_only=True,
                        mode="min",
                        verbose=1
                    )

                    earlystop_cb = EarlyStopping(
                        monitor="val_loss",
                        patience=10,  # stop if no improvement after 10 epochs
                        verbose=1,
                        # restore_best_weights=True  # only for immediate prediction
                        restore_best_weights=False
                    )

                    history = model.fit(
                        X_train, y_train,
                        validation_data=(X_val, y_val),
                        epochs=100,
                        batch_size=128,
                        callbacks=[checkpoint_cb, earlystop_cb]
                    )

                    # Making predictions
                    # predictions = model.predict(X_val)
                    #
                    # plt.figure(figsize=(1000, 6))
                    # plt.plot(y_val, label='True Prices', marker='o')
                    # plt.plot(predictions, label='Predicted Prices', marker='x')
                    # plt.title('True vs Predicted Prices')
                    # plt.xlabel('Sample Index')
                    # plt.ylabel('Price')
                    # plt.legend()
                    # plt.savefig('test.png')