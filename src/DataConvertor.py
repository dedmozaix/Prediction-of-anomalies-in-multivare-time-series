import torch
import torch.utils.data

from src.postgres_utils.db_connector import DBConnector, DBWriter
from src.BaseObjects import PredictionModelType


class MTSFDataset(torch.utils.data.Dataset):

    def __init__(self, set_type, hparams, alg_object, db_connector_obj):

        self.set_type = set_type
        self.model_type = alg_object
        self.db_connector_obj = db_connector_obj

        self.pred_window = hparams.pred_horizon
        self.train_window = hparams.train_horizon
        self.data_name = hparams.data_name
        self.window = hparams.window
        self.batch_size = hparams.batch_size

        self.n_layers = hparams.n_layers

    def set_set_type(self, set_type):
        self.set_type = set_type

    def split_data(self):
        if self.set_type is not None:
            self.split_x_y_data()
        else:
            raise NotImplementedError

    def save_model_metadata(self, metadata):
        db_writer = DBWriter(database_name="postgres", db_connector_type=self.db_connector_obj)
        db_writer.create_new_table()
        db_writer.insert_into_table(metadata)

    def split_x_y_data(self):

        db_connector = DBConnector(dataset_name=self.data_name, dataset_type=self.set_type,
                                   pred_horizon=self.pred_window, db_limit=self.train_window, db_connector_type=self.db_connector_obj)
        rawdata = torch.tensor(db_connector.get_db()).view(-1, db_connector.get_column_count())

        if self.model_type == PredictionModelType.MARIMA:
            rawdata = torch.tensor(db_connector.get_db()).view(-1, db_connector.get_column_count()).T

            self.sample_num, self.len = rawdata.shape
            self.samples = rawdata.numpy()

        elif self.model_type == PredictionModelType.DSANET:

            self.len, self.var_num = rawdata.shape
            self.sample_num = max(self.len - self.window - self.pred_window + 1, 0)
            self.samples, self.labels = self.__getsamples(rawdata)

        elif self.model_type == PredictionModelType.LSTM:

            self.len, self.var_num = rawdata.shape
            self.sample_num = self.batch_size * round((self.len - self.window - self.pred_window - 1) / self.batch_size)
            self.samples, self.labels = self.__getsamples(rawdata)

            self.samples = self.samples.numpy()
            self.labels = self.labels.numpy()

        db_connector.close_connection()

    def __getsamples(self, data):
        X = torch.zeros((self.sample_num, self.window, self.var_num))
        Y = torch.zeros((self.sample_num, 1, self.var_num))

        for i in range(self.sample_num):
            start = i
            end = i + self.window
            X[i, :, :] = data[start:end, :]
            Y[i, :, :] = data[end + self.pred_window - 1, :]
        print("GOT DATA:", X.shape, "\n", Y.shape, "\n")

        return X, Y

    def __len__(self):
        return self.sample_num

    def __getitem__(self, idx):
        if self.model_type == PredictionModelType.MARIMA:
            return self.samples[idx]
        else:
            return [self.samples[idx], self.labels[idx]]

    def get_orig_preds(self, n_time_steps):
        db_connector = DBConnector(dataset_name=self.data_name, dataset_type=self.set_type,
                                   pred_horizon=self.pred_window, db_limit=self.train_window, db_connector_type=self.db_connector_obj)
        if self.model_type == PredictionModelType.MARIMA:
            return torch.tensor(db_connector.get_preds(n_time_steps)).view(-1, db_connector.get_column_count())
        else:
            return torch.tensor(db_connector.get_preds(n_time_steps)).view(-1, db_connector.get_column_count())

    def get_n_multiv(self, data_name):
        db_connector = DBConnector(dataset_name=data_name, dataset_type='train',
                                   pred_horizon=None, db_connector_type=self.db_connector_obj)
        return db_connector.get_column_count()
