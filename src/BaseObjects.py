from prediction_models.LSTM_dir.LSTM_model import LSTM_class
from prediction_models.DSANET.DSA_model import DSANet
from prediction_models.Multivariant_ARIMA.MARIMA_model import MARIMA

#from deepod.models.dsad import DeepSAD
from pyod.models.deep_svdd import DeepSVDD
from pyod.models.copod import COPOD
#from deepod.models.icl import ICL

import enum


class PredictionModelType(enum.Enum):
    LSTM = {"name": "LSTM", "model": LSTM_class, "val_enabled": True, "test_enabled": True, "review_enabled": True}
    DSANET = {"name": "DSANET", "model": DSANet, "val_enabled": True, "test_enabled": True, "review_enabled": True}
    MARIMA = {"name": "MARIMA", "model": MARIMA, "val_enabled": False, "test_enabled": True, "review_enabled": False}


class DetectionModelType(enum.Enum):
    COPOD = {"name": "COPOD", "model": COPOD}
    DeepSVDD = {"name": "DeepSVDD", "model": DeepSVDD}

class DBConnectorType(enum.Enum):
    POSTGRES = {"host": "localhost", "user": "postgres", "password": "Cool_564", "port": "5432",
                "table_name": "model_metadata"}