from __future__ import annotations
from typing import Dict, Any, List, Tuple

import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score

try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None

from src.crispml.common.logging.logging_utils import get_logger
logger = get_logger(__name__)


def _build_regressor(name: str, hp: Dict) -> Any:
    if name == "linear": return LinearRegression(**hp)
    if name == "ridge": return Ridge(**hp)
    if name == "lasso": return Lasso(**hp)
    if name == "tree_reg": return DecisionTreeRegressor(**hp)
    if name == "rf_reg": return RandomForestRegressor(**hp)
    if name == "knn_reg": return KNeighborsRegressor(**hp)
    if name == "svr": return SVR(**hp)
    if name == "poly":
        d = hp.get("degree", 2)
        return Pipeline([("poly", PolynomialFeatures(d)), ("lin", LinearRegression())])
    if name == "xgboost":
        if XGBRegressor is None:
            raise ImportError("xgboost not installed")
        return XGBRegressor(**hp)
    if name == "nn":
        return MLPRegressor(max_iter=200, random_state=42, **hp)
    raise ValueError(f"Regressor not supported: {name}")


def run_regression_algos(
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
        algos: List[str],
        hyperparams: Dict[str, Dict],
) -> Tuple[Dict[str, Any], Dict[str, float]]:
    models = {}
    scores = {}

    for algo in algos:
        hp = hyperparams.get(algo, {})

        if any(isinstance(v, list) for v in hp.values()):
            reg = GridSearchCV(_build_regressor(algo, {}), hp, cv=3, n_jobs=-1)
            logger.info("[modeling] GridSearchCV for regressor '%s'", algo)
        else:
            reg = _build_regressor(algo, hp)
            logger.info("[modeling] Training regressor '%s'", algo)

        reg.fit(X_train, y_train)
        r2 = r2_score(y_test, reg.predict(X_test))

        models[algo] = reg
        scores[algo] = r2

        logger.info("[modeling] Regressor '%s' R2 test = %.4f", algo, r2)

    return models, scores
