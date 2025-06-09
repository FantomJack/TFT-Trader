import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lightning.pytorch as pl
import torch
from pytorch_forecasting import (TimeSeriesDataSet, GroupNormalizer,
                                 Baseline, TemporalFusionTransformer, QuantileLoss)
from lightning.pytorch.callbacks import (EarlyStopping, LearningRateMonitor, ModelCheckpoint)
from lightning.pytorch.loggers import TensorBoardLogger
from pytorch_forecasting.data import TorchNormalizer

def df_prep(tickers, folder):
    df_list = []

    for label in tickers:
        tmp = pd.read_csv(folder + label + '.csv')
        tmp.index = pd.to_datetime(tmp.index)
        tmp.sort_index(inplace=True)
        tmp['ticker'] = label
        tmp['idx'] = range(len(tmp))
        df_list.append(tmp)

    df = pd.concat(df_list).reset_index(drop=True)
    return df

class TFTModelWrapper:
    def __init__(self, target: str, covariates: list, known_reals: list, hyperparams: dict, num_workers: int = 8):
        required_keys = ["batch_size", "max_encoder_length", "max_prediction_length", "lstm_layers", "learning_rate",
                         "hidden_size", "attention_head_size", "dropout", "hidden_continuous_size", "quantiles"]
        for key in required_keys:
            if key not in hyperparams:
                raise ValueError(f"Chýba vyžadovaný hyperparameter: {key}")

        self.target = target
        self.covariates = covariates
        self.known_reals = known_reals
        self.hyperparams = hyperparams
        self.quantiles = hyperparams.get("quantiles")
        self.num_workers = num_workers

        # These will be defined in prepare_datasets.
        self.val_data = None
        self.training_dataset = None
        self.validation_dataset = None
        self.testing_dataset = None
        self.train_dataloader = None
        self.val_dataloader = None
        self.test_dataloader = None

        # Model and trainer will be created later.
        self.model = None
        self.trainer = None
        self.best_model_path = None

    def prepare_datasets(self, train_data: pd.DataFrame, val_data: pd.DataFrame, test_data: pd.DataFrame = None):
        """Creates TimeSeriesDataSet objects and their corresponding dataloaders."""
        max_encoder_length = self.hyperparams.get("max_encoder_length", 30)
        max_prediction_length = self.hyperparams.get("max_prediction_length", 3)

        # Build training dataset using all provided training data.
        self.training_dataset = TimeSeriesDataSet(
            train_data,
            time_idx="idx",
            target=self.target,
            group_ids=['ticker'],
            static_categoricals=['ticker'],
            min_encoder_length=max_encoder_length // 2,
            max_encoder_length=max_encoder_length,
            min_prediction_length=1,
            max_prediction_length=max_prediction_length,
            time_varying_known_reals=self.known_reals,
            time_varying_unknown_reals=self.covariates,
            target_normalizer=GroupNormalizer(method="standard", groups=["ticker"]),
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True
        )
        # Build validation dataset from the training dataset
        self.val_data = val_data
        self.validation_dataset = TimeSeriesDataSet.from_dataset(
            self.training_dataset, val_data, predict=False, stop_randomization=True
        )

        # If test data is provided, build the testing dataset.
        if test_data is not None:
            self.testing_dataset = TimeSeriesDataSet.from_dataset(
                self.training_dataset, test_data, predict=False, stop_randomization=True
            )

        batch_size = self.hyperparams.get("batch_size")
        self.train_dataloader = self.training_dataset.to_dataloader(persistent_workers=True, train=True, batch_size=batch_size,
                                                                    num_workers=self.num_workers)
        self.val_dataloader = self.validation_dataset.to_dataloader(persistent_workers=True, train=False, batch_size=batch_size * 10,
                                                                    num_workers=self.num_workers)
        if test_data is not None:
            self.test_dataloader = self.testing_dataset.to_dataloader(persistent_workers=True, train=False, batch_size=batch_size * 10,
                                                                      num_workers=self.num_workers)

    def build_model(self):
        """Builds the TFT model from the training dataset."""
        self.model = CustomTFT.from_dataset(
            self.training_dataset,
            lstm_layers=self.hyperparams.get("lstm_layers"),
            learning_rate=self.hyperparams.get("learning_rate", 0.0005),
            hidden_size=self.hyperparams.get("hidden_size"),
            attention_head_size=self.hyperparams.get("attention_head_size"),
            dropout=self.hyperparams.get("dropout"),
            hidden_continuous_size=self.hyperparams.get("hidden_continuous_size"),
            output_size=len(self.quantiles),
            loss=QuantileLoss(quantiles=self.quantiles),
            log_interval=self.hyperparams.get("log_interval", 10),
            reduce_on_plateau_patience=self.hyperparams.get("reduce_on_plateau_patience", 4)
        )

    def train(self, max_epochs: int = 1000, patience: int = 30, dirpath: str = './checkpoints/', top_k: int = 1, use_gpu: bool = True):
        """
        Trains the model using PyTorch Lightning.
        """
        self.build_model()

        torch.set_float32_matmul_precision('medium')
        early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=patience, verbose=True,
                                            mode="min")
        lr_logger = LearningRateMonitor()
        logger = TensorBoardLogger("lightning_logs")
        checkpoint_callback = ModelCheckpoint(
            monitor='val_loss',
            mode='min',
            save_top_k=top_k,
            dirpath=dirpath,
            filename='model-{epoch:02d}-{val_loss:.2f}'
        )
        self.trainer = pl.Trainer(
            max_epochs=max_epochs,
            accelerator='gpu' if use_gpu else 'cpu',
            devices=1 if use_gpu else None,
            enable_model_summary=True,
            gradient_clip_val=0.05,
            callbacks=[lr_logger, early_stop_callback, checkpoint_callback],
            logger=logger
        )
        self.trainer.fit(
            self.model,
            train_dataloaders=self.train_dataloader,
            val_dataloaders=self.val_dataloader
        )
        self.best_model_path = self.trainer.checkpoint_callback.best_model_path
        print("Best model path: ", self.best_model_path)
        self.model = TemporalFusionTransformer.load_from_checkpoint(self.best_model_path)

    def load_checkpoint(self, model_path):
        self.model = TemporalFusionTransformer.load_from_checkpoint(model_path)

    def predict_validation(self, plot_results: bool = False):
        """Generates raw predictions on the validation set."""
        raw_predictions = self.model.predict(self.val_dataloader, mode="raw", return_x=True,
                                             num_workers=self.num_workers)
        if plot_results:
            self.model.plot_prediction(raw_predictions.x, raw_predictions.output, idx=1, add_loss_to_title=True)
            plt.savefig('out1.png')

            predictions = raw_predictions.output.prediction
            self.multi_plot(predictions, self.val_data,[0,100,200,300,900],100, 'val',True)

            # output_length = len(predictions)
            # x = np.arange(0, output_length)
            # rewards0 = []
            #
            # index = 0
            # for prediction in range(output_length):
            #     i = predictions[prediction, index]
            #     rewards0.append(i)


            # self.plot_data(0, 100, rewards0, x, y, "val" + str(index) + "_100")
            # self.plot_data(100, 200, rewards0, x, y, "val" + str(index) + "_200")
            # self.plot_data(200, 300, rewards0, x, y, "val" + str(index) + "_300")
            # self.plot_data(300, 400, rewards0, x, y, "val" + str(index) + "_400")
            # self.plot_data(900, 1000, rewards0, x, y, "val" + str(index) + "_1000")


        return raw_predictions

    def predict_test(self, test_data: pd.DataFrame = None):
        """Generates raw predictions on the test set."""

        if self.test_dataloader is None and test_data is None:
            raise ValueError("No test data provided.")

        if test_data is not None:
            self.testing_dataset = TimeSeriesDataSet.from_dataset(
                self.training_dataset, test_data, predict=False, stop_randomization=True
            )
            self.test_dataloader = self.testing_dataset.to_dataloader(train=False, persistent_workers=True,
                                                                      batch_size=batch_size * 10,
                                                                      num_workers=self.num_workers)

        return self.model.predict(self.test_dataloader, mode="raw", return_x=True, num_workers=self.num_workers)

    def multi_plot(self, predictions, values, plot_idxs: list[int], plot_len: int, prefix: str = 'plot', first_step_only: bool = False):

        pred_len = self.hyperparams.get("max_prediction_length")
        encoder_size = self.hyperparams.get("max_encoder_length")
        prediction_steps = np.empty(pred_len, dtype=object)

        for step in range(pred_len):
            prediction_steps[step] = predictions[:, step, int(len(self.quantiles) / 2)].cpu()
            prediction_steps[step] = torch.cat((torch.tensor([np.nan] * step), prediction_steps[step]))

        x_all = np.arange(0, len(predictions))
        y0 = values[self.target].iloc[encoder_size:]
        y_all = y0.values

        for idx in plot_idxs:
            x = x_all[idx: idx + plot_len]
            y = y_all[idx: idx + plot_len]

            plt.cla()
            plt.figure(figsize=(15, 5))

            if first_step_only:
                plt.plot(x, prediction_steps[0][idx: idx + plot_len], label='predicted-0.5')
            else:
                for step in range(pred_len):
                    plt.plot(x, prediction_steps[step][idx: idx + plot_len], label='step-' + str(step))

            plt.plot(x, y, label='observed')
            plt.axhline(y=2, color='tab:gray', linestyle='-')
            plt.axhline(y=-2, color='tab:gray', linestyle='-')
            plt.legend()
            plt.savefig(prefix + str(idx) + '.png')

    def plot_data(self, start, end, predictions, x, y, name):
        median = int(len(self.quantiles)/2)
        # reward0 = np.array([tensor[0].numpy() for tensor in rewards0])[start:end]
        reward2 = np.array([tensor[median].item() for tensor in predictions])[start:end]
        print(reward2)

        y_slice = y[start:end]
        x_slice = x[start:end]

        plt.cla()
        plt.figure(figsize=(15, 5))  # Set the figure size
        plt.plot(x_slice, reward2, label='predicted-0.5')
        plt.plot(x_slice, y_slice, label='observed')
        plt.axhline(y=2, color='tab:gray', linestyle='-')
        plt.axhline(y=-2, color='tab:gray', linestyle='-')
        plt.axhline(y=1, color='tab:gray', linestyle='-')
        plt.axhline(y=-1, color='tab:gray', linestyle='-')
        plt.legend()  # Don't forget to call plt.legend()

        plt.savefig(name + '.png')


class CustomTFT(TemporalFusionTransformer):
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=0.0005, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=10, factor=0.5, verbose=True
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}

