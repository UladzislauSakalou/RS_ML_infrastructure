from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_validate, GridSearchCV
import mlflow
from typing import Any
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from joblib import dump
import click
from typing import Tuple


def nestedCV(
    model_name: str,
    features: pd.DataFrame,
    target: pd.Series,
    random_state: int,
    scoring: str,
    n_splits: int = 5,
    k_splits: int = 2,
) -> Tuple[float, float, float]:
    cv_inner = KFold(n_splits=k_splits, shuffle=True, random_state=4)
    model = get_model(model_name, random_state)
    param_grid = get_param_grid(model_name)
    search = GridSearchCV(
        model, param_grid, scoring=scoring, n_jobs=-1, cv=cv_inner, refit=True
    )
    cv_outer = KFold(n_splits=n_splits, shuffle=True, random_state=4)
    scores = cross_validate(
        search,
        features,
        target,
        scoring=["accuracy", "precision_macro", "f1_macro"],
        cv=cv_outer,
        n_jobs=-1,
    )
    return (
        np.mean(scores["test_accuracy"]),
        np.mean(scores["test_precision_macro"]),
        np.mean(scores["test_f1_macro"]),
    )


def get_tuned_model(
    model_name: str,
    features: pd.DataFrame,
    target: pd.Series,
    random_state: int,
    scoring: str = "accuracy",
    n_splits: int = 5,
) -> Any:
    k_fold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    model = get_model(model_name, random_state)
    param_grid = get_param_grid(model_name)
    search = GridSearchCV(
        model, param_grid, scoring=scoring, n_jobs=-1, cv=k_fold, refit=True
    )
    search.fit(features, target)
    mlflow.log_params(search.best_params_)
    return search.best_estimator_


def get_model(model_name: str, random_state: int) -> Any:
    if model_name == "rf":
        return RandomForestClassifier(random_state=random_state)
    elif model_name == "lr":
        return LogisticRegression(random_state=random_state)


def get_param_grid(model_name: str) -> dict[str, Any]:
    param_grid: dict[str, Any] = dict()
    if model_name == "rf":
        param_grid["n_estimators"] = [100, 200]
        param_grid["criterion"] = ["gini", "entropy"]
        param_grid["max_depth"] = [None, 5, 10]
        param_grid["min_samples_split"] = [2, 4]
        param_grid["min_samples_leaf"] = [2, 4]
    elif model_name == "lr":
        param_grid["C"] = [0.1, 1, 10]
        param_grid["max_iter"] = [100, 500]
    return param_grid


def save_model(model: Any, save_model_path: Path) -> None:
    dump(model, save_model_path)
    click.echo(f"Model is saved to {save_model_path}.")
