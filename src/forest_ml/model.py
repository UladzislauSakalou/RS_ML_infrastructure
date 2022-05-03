import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
import mlflow
from typing import Any

def nestedCV(
    model_name: str,
    pipeline: Pipeline,
    features: pd.DataFrame,
    target: pd.Series,
    scoring: str,
    n_splits: int = 5,
    k_splits: int = 2,
) -> float:
    cv_inner = KFold(n_splits=k_splits, shuffle=True, random_state=4)
    param_grid = get_param_grid(model_name)
    search = GridSearchCV(
        pipeline, param_grid, scoring=scoring, n_jobs=-1, cv=cv_inner, refit=True
    )
    cv_outer = KFold(n_splits=n_splits, shuffle=True, random_state=4)
    scores = cross_val_score(
        search, features, target, scoring=scoring, cv=cv_outer, n_jobs=-1
    )
    return np.mean(scores)


def get_tuned_model(
    model_name: str,
    pipeline: Pipeline,
    features: pd.DataFrame,
    target: pd.Series,
    scoring: str = "accuracy",
    n_splits: int = 5,
):
    k_fold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    param_grid = get_param_grid(model_name)
    search = GridSearchCV(
        pipeline, param_grid, scoring=scoring, n_jobs=-1, cv=k_fold, refit=True
    )
    search.fit(features, target)
    mlflow.log_params(search.best_params_)
    return search.best_estimator_


def get_param_grid(model_name: str):
    param_grid: dict[str, Any] = dict()
    if model_name == "rf":
        param_grid["classifier__n_estimators"] = [50, 100, 200]
        param_grid["classifier__criterion"] = ["gini", "entropy"]
        param_grid["classifier__max_depth"] = [None, 5, 10]
        param_grid["classifier__min_samples_split"] = [2, 4]
        param_grid["classifier__min_samples_leaf"] = [2, 4]
    elif model_name == "lr":
        param_grid["classifier__C"] = [0.1, 1, 10]
        param_grid["classifier__max_iter"] = [100, 500]
    return param_grid