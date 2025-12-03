from __future__ import annotations
import pandas as pd
from typing import Dict, Any, List

try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.statespace.sarimax import SARIMAX
except ImportError:
    ARIMA = None
    SARIMAX = None

from src.crispml.common.logging.logging_utils import get_logger
logger = get_logger(__name__)


def run_time_series_algos(
        y_train: pd.Series,
        algos: List[str],
        hyperparams: Dict[str, Dict],
) -> Dict[str, Any]:

    if ARIMA is None or SARIMAX is None:
        raise ImportError("statsmodels is required for ARIMA/SARIMA")

    models = {}

    for algo in algos:
        hp = hyperparams.get(algo, {})

        if algo == "arima":
            order = (hp.get("p", 1), hp.get("d", 0), hp.get("q", 0))
            model = ARIMA(y_train, order=order).fit()
            models["arima"] = model
            logger.info("[modeling] ARIMA trained with order=%s", order)

        elif algo == "sarima":
            order = (hp.get("p", 1), hp.get("d", 0), hp.get("q", 0))
            seasonal = (hp.get("P", 0), hp.get("D", 0), hp.get("Q", 0), hp.get("m", 12))
            model = SARIMAX(y_train, order=order, seasonal_order=seasonal).fit()
            models["sarima"] = model
            logger.info("[modeling] SARIMA trained with order=%s seasonal=%s",
                        order, seasonal)

        else:
            logger.warning("[modeling] Unknown time-series algo: %s", algo)

    return models
