from collections import OrderedDict
from torch.utils.data import DataLoader
import pytorch_lightning as pl

import torch
import torch.nn as nn
from torch import optim


class LSTM_class(pl.LightningModule):

    def __init__(self, hparams, reader):
        super(LSTM_class, self).__init__()
        # self.hparams = hparams

        self.batch_size = hparams.batch_size

        # parameters from dataset
        self.window = hparams.window
        self.local = hparams.local
        self.n_multiv = reader.get_n_multiv(hparams.data_name)
        self.n_kernels = hparams.n_kernels
        self.w_kernel = hparams.w_kernel
        self.train_horizon = hparams.train_horizon

        # hyperparameters of model
        self.d_model = hparams.d_model
        self.d_inner = hparams.d_inner
        self.n_layers = hparams.n_layers
        self.n_head = hparams.n_head
        self.d_k = hparams.d_k
        self.d_v = hparams.d_v
        self.drop_prob = hparams.drop_prob

        self.learning_rate = hparams.learning_rate
        self.window = hparams.window
        self.data_name = hparams.data_name
        self.criterion = hparams.criterion
        self.pred_horizon = hparams.pred_horizon

        self.reader_obj = reader

        self.preds_list = []
        self.original_preds_list = []
        self.train_history = []
        self.val_history = []
        self.test_history = []

        # build model
        self.__build_model()

    def get_parameters(self):
        return self.parameters()

    def __build_model(self):

        self.lstm = nn.LSTM(num_layers=self.n_layers, input_size=self.window, hidden_size=self.n_multiv,
                            batch_first=True, dropout=self.drop_prob, bidirectional=False)
        #print("\nLSTM: ", self.lstm.input_size, self.lstm.hidden_size)
        self.fc_1 = nn.Linear(self.n_multiv, self.n_multiv)
        self.fc_2 = nn.Linear(self.n_multiv, 1)
        self.relu = nn.ReLU()

    def forward(self, x, predict=False):
        hidden = (torch.zeros(self.n_layers, x.shape[0], self.n_multiv),
                  torch.zeros(self.n_layers, x.shape[0], self.n_multiv))

        output, (hn, cn) = self.lstm(x, hidden)
        out = self.fc_1(output[:, 0, :])  # Selecting the last output
        print("OUT: ", out.shape)


        return out

    def training_step(self, data_batch, batch_i):
        print('\nTraining step called')
        x, y = data_batch
        y = torch.squeeze(y)
        print("TRAIN: ", x.shape, y.shape)
        y_hat = self.forward(x)
        loss_val = self.loss(y, y_hat)

        self.train_history.append(loss_val.detach().cpu().numpy())
        output = OrderedDict({
            'loss': loss_val,
            'y': y,
            'y_hat': y_hat,
        })
        return output

    def validation_step(self, data_batch, batch_i):
        print('\nValidation step called')
        x, y = data_batch
        print("VALIDATION: ", x.shape, y.shape)
        y_hat = self.forward(x)
        loss_val = self.loss(y, y_hat)

        self.val_history.append(loss_val.detach().cpu().numpy())
        output = OrderedDict({
            'loss': loss_val,
            'y': y,
            'y_hat': y_hat,
        })
        return output

    def test_step(self, data_batch):
        print('\nTest step called')
        x, y = data_batch
        y_hat = self.forward(x)
        loss_val = self.loss(y, y_hat)

        self.test_history.append(loss_val.detach().cpu().numpy())
        output = OrderedDict({
            'loss': loss_val,
            'y': y,
            'y_hat': y_hat,
        })
        return output

    def predict_step(self, data_batch):
        x, y = data_batch
        preds = self.forward(x)

        # self.preds_list.append(y_hat)
        self.original_preds_list.append(y)

        return preds

    def get_orig_preds(self):
        return self.original_preds_list

    def loss(self, labels, predictions):
        if self.criterion == "mae":
            criterion = torch.nn.L1Loss()
        else:
            criterion = torch.nn.MSELoss()

        loss = criterion(predictions, labels)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.lstm.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        return [optimizer], [scheduler]

    def __dataloader(self, train):
        self.reader_obj.set_set_type(set_type=train)
        self.reader_obj.split_data()

        loader = DataLoader(
            dataset=self.reader_obj,
            batch_size=self.batch_size,
            shuffle=True if train == "train" else False,
            persistent_workers=True,
            num_workers=15
        )
        return loader

    def train_dataloader(self):
        print('traing data loader called')
        return self.__dataloader(train='train')

    def val_dataloader(self):
        print('val data loader called')
        return self.__dataloader(train='validation')

    def test_dataloader(self):
        print('test data loader called')
        return self.__dataloader(train='test')
