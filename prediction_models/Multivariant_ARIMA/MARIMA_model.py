from torch.utils.data import DataLoader
import pytorch_lightning as pl
import numpy as np
import pmdarima as pm


class MARIMA(pl.LightningModule):

    def __init__(self, hparams, reader):
        super(MARIMA, self).__init__()

        self.models = []

        self.idj_maximum = hparams.idj_maximum
        self.pred_horizon = hparams.pred_horizon

        # parameters from dataset
        self.data_name = hparams.data_name
        self.train_horizon = hparams.train_horizon

        self.reader_obj = reader

    def get_parameters(self):
        return self.parameters()

    def forward(self, x):
        # model = pm.auto_arima(x, d=None, start_p=1, start_q=1, max_p=self.idj_maximum, max_q=self.idj_maximum)
        model = pm.auto_arima(x, start_p=0, start_q=0,
                              max_p=self.idj_maximum, max_q=self.idj_maximum, m=1,
                              start_P=0, seasonal=False,
                              d=None, D=1, trace=False,
                              error_action='ignore',  # we don't want to know if an order does not work
                              suppress_warnings=True,  # we don't want convergence warnings
                              stepwise=True)
        # print(model.summary())
        return model

    def training_step(self, data_batch, batch_i):
        tmp_mdl = self.forward(data_batch[0].tolist())
        self.models.append(tmp_mdl)

        return None

    def predict_step(self, data_batch, batch_i):
        return self.models[batch_i].predict(n_periods=self.pred_horizon)

    def get_orig_preds(self):
        return self.reader_obj.get_orig_preds()

    def __dataloader(self, type):
        self.reader_obj.set_set_type(set_type=type)
        self.reader_obj.split_data()

        loader = DataLoader(
            dataset=self.reader_obj,
            batch_size=1,
            shuffle=False,
            num_workers=15,
            persistent_workers=True
        )

        return loader

    def train_dataloader(self):
        print('traing data loader called')
        return self.__dataloader(type='train')

    def test_dataloader(self):
        print('test data loader called')
        return self.__dataloader(type='test')

    def configure_optimizers(self):
        return None
