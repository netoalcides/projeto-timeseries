from typing import Tuple, Dict
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error, mean_squared_error

# import timesfm
from timesfm import TimesFm, TimesFmHparams, TimesFmCheckpoint

FORECAST_HORIZON = 12
MIN_CONTEXT_LEN = 256

tfm = TimesFm(
    hparams=TimesFmHparams(
        backend="cpu",
        per_core_batch_size=1,
        horizon_len=FORECAST_HORIZON,
        num_layers=16,
        context_len=512,
        use_positional_embedding=False,
    ),
    checkpoint=TimesFmCheckpoint(
        huggingface_repo_id="google/timesfm-1.0-200m"        
    ),
)
print("Modelo carregado")