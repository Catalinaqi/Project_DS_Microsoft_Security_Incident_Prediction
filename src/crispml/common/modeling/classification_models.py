from __future__ import annotations
import numpy as np
from typing import Dict, Any, List, Tuple

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

from src.crispml.common.logging.logging_utils import get_logger

logger = get_logger(__name__)


def _build_classifier(name: str, hp: Dict) -> Any:
    if name == "dt":
        return DecisionTreeClassifier(**hp)
    if name == "rf":
        return RandomForestClassifier(**hp)
    if name == "svm":
        return SVC(probability=True, **hp)
    if name == "nb":
        return GaussianNB(**hp)
    if name == "knn":
        return KNeighborsClassifier(**hp)
    raise ValueError(f"Classifier not supported: {name}")


def run_classification_algos(
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
        algos: List[str],
        hyperparams: Dict[str, Dict],
) -> Tuple[Dict[str, Any], Dict[str, float]]:
    """
    Trains multiple classifiers.
    Returns: models dict + accuracy dict
    """
    models = {}
    scores = {}

    for algo in algos:
        hp = hyperparams.get(algo, {})

        if any(isinstance(v, list) for v in hp.values()):
            base_model = _build_classifier(algo, {})
            clf = GridSearchCV(base_model, hp, cv=3, n_jobs=-1)
            logger.info("[modeling] GridSearchCV for classifier '%s'", algo)
        else:
            clf = _build_classifier(algo, hp)
            logger.info("[modeling] Training classifier '%s'", algo)

        clf.fit(X_train, y_train)
        acc = accuracy_score(y_test, clf.predict(X_test))

        models[algo] = clf
        scores[algo] = acc

        logger.info("[modeling] Classifier '%s' accuracy test = %.4f", algo, acc)

    return models, scores
