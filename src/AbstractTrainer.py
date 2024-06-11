from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
import numpy as np
import torch
import pytorch_lightning as pl

import time
import psutil
import os

from src.DataConvertor import MTSFDataset
from matplotlib import pyplot as plt

class AbstractModelTrainer(pl.LightningModule):
    def __init__(self, hparams, alg_type, detector_type, db_connector_obj):
        super(AbstractModelTrainer, self).__init__()

        self.alg_object = alg_type
        self.detector_object = detector_type
        self.db_connector_obj = db_connector_obj

        self.data_name = hparams.data_name
        self.pred_window = hparams.pred_horizon
        self.num_epochs = hparams.num_epochs
        self.data_reader = MTSFDataset(None, hparams, self.alg_object, db_connector_obj)

        self.trainer = pl.Trainer(
            accelerator="cpu",
            max_epochs=hparams.num_epochs,
            use_distributed_sampler=True,
            enable_checkpointing=False
        )

        self.mae, self.smape, self.mape = [], [], []
        self.trainig_time, self.pred_time, self.detection_time = 0, 0, 0
        self.used_memory, self.cpu_times, self.disk_usage = 0, 0, 0

        if hparams.pretrained_model is True:
            self.model = torch.load(hparams.pretrained_model_path)
        else:
            self.model = self.alg_object.value["model"](hparams, self.data_reader)
        self.anomaly_detector = self.detector_object.value["model"]()

    def get_predictions(self):
        return self.predictions

    def get_original_predictions(self):
        return self.original_preds

    def get_proba_predictions(self):
        return self.proba_predictions

    def get_metrics_report(self):
        return (self.num_epochs, self.mae, self.smape, self.mape,
                self.trainig_time, self.pred_time, self.detection_time, self.num_parameters,
                psutil.cpu_count(), self.cpu_times, self.disk_usage, self.used_memory)

    def fit(self):
        print("\n GOING TO FIT")
        print("Used CPUs", psutil.cpu_percent(percpu=True))
        self.cpu_times = sum(psutil.cpu_percent(percpu=True))
        print("CPU times", psutil.cpu_times())
        print("Disk usage", psutil.disk_usage('/'))
        self.disk_usage = psutil.disk_usage('/').percent
        print('memory % used:', psutil.virtual_memory()[2])
        self.used_memory = psutil.virtual_memory()[2]


        start_time = time.time()
        self.__execute_training()
        self.trainig_time = time.time() - start_time

        self.predictions, self.original_preds, self.model_parameters = self.__execute_prediction()
        self.pred_time = time.time() - self.trainig_time

        self.num_parameters = sum(p.numel() for p in self.model_parameters)

        """for i in range(0, 20):
            plt.plot(self.original_preds.T[i], label="original")
            plt.plot(self.predictions.T[i], label="predicted")
            plt.legend()
            plt.show()
        print("PARAMETERS = ", self.model_parameters, "count", sum(p.numel() for p in self.model_parameters))"""

        self.__execute_anomaly_detection()
        self.detection_time = time.time() - self.pred_time

    def __execute_training(self):
        print("\n GOING TO TRAIN")
        if self.alg_object.value["val_enabled"]:
            self.trainer.fit(self.model, self.model.train_dataloader(), self.model.val_dataloader())
        else:
            self.trainer.fit(self.model, self.model.train_dataloader())

    def __execute_prediction(self):
        print("\n GOING TO PREDICT")
        if self.alg_object.value["test_enabled"]:
            self.trainer.fit(self.model, self.model.test_dataloader())

        predictions = self.trainer.predict(self.model, self.model.test_dataloader())

        if self.alg_object.value["review_enabled"]:
            predictions = torch.concatenate(predictions, dim=0)  # concat predictions
            predictions = predictions[:-1]  # remove last idx which was artificially added
            predictions = torch.squeeze(predictions)  # remove extra dimension of size 1
        else:
            predictions = torch.tensor(predictions).T

        original_data = self.data_reader.get_orig_preds(predictions.shape[0])
        print("\n PREDICTIONS = ", predictions.shape, "\n ORIGINAL DATA = ", original_data.shape)

        return predictions, original_data, self.model.get_parameters()

    def __execute_anomaly_detection(self):
        print("ANOMALY DETECTION")

        self.anomaly_detector.fit(np.array(self.predictions)) #, np.array(self.original_preds))
        self.proba_predictions = self.anomaly_detector.decision_function(np.array(self.predictions))
        # print("Predictions proba: ", proba_predictions)

    def calculate_metrics(self):
        print("CALCULATING METRICS")

        # adding 1 to predictions and original preds for correct mape
        self.predictions.add_(1)
        self.original_preds.add_(1)

        try:
            for batch in range(0, self.pred_window):
                self.mae.append(float(mean_absolute_error(self.original_preds.T[batch], self.predictions.T[batch])))
                self.smape.append(self.calculate_smape(self.original_preds.T[batch], self.predictions.T[batch]))
                self.mape.append(
                    float(mean_absolute_percentage_error(self.original_preds.T[batch], self.predictions.T[batch])))

        except Exception as e:
            print("Got error while calculating metrics", e)

        # subtracting 1 from predictions and original preds for balance
        self.predictions.sub_(1)
        self.original_preds.sub_(1)

    def calculate_smape(self, preds, orig):
        tmp = 2 * np.abs(preds - orig) / (np.abs(orig) + np.abs(preds))
        len_ = np.count_nonzero(~np.isnan(tmp))
        if len_ == 0 or np.nansum(tmp) == 0:  # and
            smape = 100
        else:
            smape = 100 / len_ * np.nansum(tmp)
        return float(smape)

    def save_report(self):
        print("SAVING REPORT")
        self.data_reader.save_model_metadata((self.data_name,
                                              self.alg_object.value["name"],
                                              self.detector_object.value["name"],
                                              self.pred_window,
                                              self.num_parameters,
                                              self.num_epochs,
                                              np.mean(self.mae),
                                              np.mean(self.smape),
                                              np.mean(self.mape),
                                              self.trainig_time,
                                              self.pred_time,
                                              self.detection_time,
                                              psutil.cpu_count(),
                                              self.cpu_times,
                                              self.disk_usage,
                                              self.used_memory))

    def save_to_file_model(self, PATH):
        torch.save(self.model, PATH)

    # AbstractTrainer.py --alg_type MARIMA --data_name HAI
    # AbstractTrainer.py --alg_type DSANET --data_name HAI
    # AbstractTrainer.py --alg_type LSTM --data_name HAI
    # ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
    # torchaudio 2.2.1+cu121 requires torch==2.2.1+cu121, but you have torch 1.13.0 which is incompatible.
    # torchvision 0.17.1 requires torch==2.2.1, but you have torch 1.13.0 which is incompatible.
