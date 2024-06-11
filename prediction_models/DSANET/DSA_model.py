from collections import OrderedDict
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import pytorch_lightning as pl

from prediction_models.DSANET.Layers import EncoderLayer


class Single_Global_SelfAttn_Module(nn.Module):

    def __init__(
            self,
            window, n_multiv, n_kernels, w_kernel,
            d_k, d_v, d_model, d_inner,
            n_layers, n_head, drop_prob=0.1):

        super(Single_Global_SelfAttn_Module, self).__init__()

        self.window = window
        self.w_kernel = w_kernel
        self.n_multiv = n_multiv
        self.d_model = d_model
        self.drop_prob = drop_prob

        self.conv2 = nn.Conv2d(1, n_kernels, (window, w_kernel))
        self.in_linear = nn.Linear(n_kernels, d_model)
        self.out_linear = nn.Linear(d_model, n_kernels)

        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=drop_prob)
            for _ in range(n_layers)])

    def forward(self, x, return_attns=False):

        x = x.view(-1, self.w_kernel, self.window, self.n_multiv)

        x2 = F.relu(self.conv2(x))
        x2 = nn.Dropout(p=self.drop_prob)(x2)
        x = torch.squeeze(x2, 2)
        x = torch.transpose(x, 1, 2)
        src_seq = self.in_linear(x)
        enc_slf_attn_list = []

        enc_output = src_seq

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output)
            if return_attns:
                enc_slf_attn_list += [enc_slf_attn]

        if return_attns:
            return enc_output, enc_slf_attn_list
        enc_output = self.out_linear(enc_output)
        return enc_output,


class Single_Local_SelfAttn_Module(nn.Module):

    def __init__(
            self,
            window, local, n_multiv, n_kernels, w_kernel,
            d_k, d_v, d_model, d_inner,
            n_layers, n_head, drop_prob=0.1):

        super(Single_Local_SelfAttn_Module, self).__init__()

        self.window = window
        self.w_kernel = w_kernel
        self.n_multiv = n_multiv
        self.d_model = d_model
        self.drop_prob = drop_prob
        self.conv1 = nn.Conv2d(1, n_kernels, (local, w_kernel))
        self.pooling1 = nn.AdaptiveMaxPool2d((1, n_multiv))
        self.in_linear = nn.Linear(n_kernels, d_model)
        self.out_linear = nn.Linear(d_model, n_kernels)

        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=drop_prob)
            for _ in range(n_layers)])

    def forward(self, x, return_attns=False):

        x = x.view(-1, self.w_kernel, self.window, self.n_multiv)

        x1 = F.relu(self.conv1(x))
        x1 = self.pooling1(x1)
        x1 = nn.Dropout(p=self.drop_prob)(x1)
        x = torch.squeeze(x1, 2)

        x = torch.transpose(x, 1, 2)
        src_seq = self.in_linear(x)

        enc_slf_attn_list = []

        enc_output = src_seq

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output)
            if return_attns:
                enc_slf_attn_list += [enc_slf_attn]

        if return_attns:
            return enc_output, enc_slf_attn_list
        enc_output = self.out_linear(enc_output)
        return enc_output,


class AR(nn.Module):

    def __init__(self, window):
        super(AR, self).__init__()
        self.linear = nn.Linear(window, 1)

    def forward(self, x):
        x = torch.transpose(x, 1, 2)
        x = self.linear(x)
        x = torch.transpose(x, 1, 2)
        return x


class DSANet(pl.LightningModule):

    def __init__(self, hparams, reader):
        super(DSANet, self).__init__()

        self.batch_size = hparams.batch_size

        # parameters from dataset
        self.window = hparams.window
        self.local = hparams.local
        self.n_multiv = reader.get_n_multiv(hparams.data_name)
        self.n_kernels = hparams.n_kernels
        self.w_kernel = hparams.w_kernel

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

        # self.tmp_max_shape = 1
        self.preds_list = []
        self.original_preds_list = []
        # build model
        self.__build_model()

    def get_parameters(self):
        return self.parameters()

    def __build_model(self):
        self.sgsf = Single_Global_SelfAttn_Module(
            window=self.window, n_multiv=self.n_multiv, n_kernels=self.n_kernels,
            w_kernel=self.w_kernel, d_k=self.d_k, d_v=self.d_v, d_model=self.d_model,
            d_inner=self.d_inner, n_layers=self.n_layers, n_head=self.n_head, drop_prob=self.drop_prob)

        self.slsf = Single_Local_SelfAttn_Module(
            window=self.window, local=self.local, n_multiv=self.n_multiv, n_kernels=self.n_kernels,
            w_kernel=self.w_kernel, d_k=self.d_k, d_v=self.d_v, d_model=self.d_model,
            d_inner=self.d_inner, n_layers=self.n_layers, n_head=self.n_head, drop_prob=self.drop_prob)

        self.ar = AR(window=self.window)
        self.W_output1 = nn.Linear(2 * self.n_kernels, 1)
        self.W_output2 = nn.Linear(1, int(self.n_kernels / 2))

        self.dropout = nn.Dropout(p=self.drop_prob)
        self.active_func = nn.Tanh()

    def forward(self, x):
        sgsf_output, *_ = self.sgsf(x)
        slsf_output, *_ = self.slsf(x)

        sf_output = torch.cat((sgsf_output, slsf_output), 2)
        sf_output = self.dropout(sf_output)
        sf_output = self.W_output1(sf_output)

        sf_output = torch.transpose(sf_output, 1, 2)

        ar_output = self.ar(x)

        output = sf_output + ar_output

        return output

    def loss(self, labels, predictions):
        loss = None
        if self.criterion == 'l1':
            loss = F.l1_loss(predictions, labels)
        elif self.criterion == 'mse':
            loss = F.mse_loss(predictions, labels)
        return loss

    def training_step(self, data_batch, batch_i):
        # forward pass
        x, y = data_batch
        y_hat = self.forward(x)

        # calculate loss
        loss_val = self.loss(y, y_hat)
        loss_val = loss_val.unsqueeze(0)

        output = OrderedDict({
            'loss': loss_val,
            'y': y,
            'y_hat': y_hat,
        })

        return output

    def validation_step(self, data_batch, batch_i):
        x, y = data_batch

        y_hat = self.forward(x)

        loss_val = self.loss(y, y_hat)

        loss_val = loss_val.unsqueeze(0)

        output = OrderedDict({
            'loss': loss_val,
            'y': y,
            'y_hat': y_hat,
        })
        return output

    def test_step(self, data_batch, batch_i):

        x, y = data_batch

        y_hat = self.forward(x)
        loss_val = self.loss(y, y_hat)
        loss_val = loss_val.unsqueeze(0)

        output = OrderedDict({
            'loss': loss_val,
            'y': y,
            'y_hat': y_hat,
        })
        return output

    def predict_step(self, data_batch, batch_i):

        x, y = data_batch
        preds = self.forward(x)

        # self.preds_list.append(preds)
        self.original_preds_list.append(y)
        return preds

    def get_orig_preds(self):
        return self.original_preds_list

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        return [optimizer], [scheduler]

    def __dataloader(self, train):
        self.reader_obj.set_set_type(set_type=train)
        self.reader_obj.split_data()

        train_sampler = None
        batch_size = self.batch_size

        try:
            if self.on_gpu:
                train_sampler = DistributedSampler(self.reader_obj, rank=self.trainer.proc_rank)
                batch_size = batch_size // self.trainer.world_size
        except Exception as e:
            pass

        should_shuffle = train_sampler is None
        loader = DataLoader(
            dataset=self.reader_obj,
            batch_size=batch_size,
            shuffle=True if train == "train" else False,
            sampler=train_sampler,
            num_workers=15,
            persistent_workers=True
        )

        return loader

    def train_dataloader(self):
        print('\n traing data loader called')
        # pl.LightningDataModule
        return self.__dataloader(train='train')

    def val_dataloader(self):
        print('\n val data loader called')
        return self.__dataloader(train='validation')

    def test_dataloader(self):
        print('\n test data loader called')
        return self.__dataloader(train='test')
