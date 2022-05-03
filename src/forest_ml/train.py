from pathlib import Path
from typing import Any
import click
from joblib import dump
from .data import get_dataset
from .pipeline import create_pipeline
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
import mlflow
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline


@click.command()
@click.option(
    "-d",
    "--dataset_path",
    default="data/train.csv",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to the dataset",
    show_default=True,
)
@click.option(
    "-s",
    "--save-model-path",
    default="data/model.joblib",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    help="Path to the model to save",
    show_default=True,
)
@click.option(
    "--model-name",
    default="rf",
    type=click.Choice(["rf", "lr"]),
    help="Model for evaluation",
    show_default=True,
)
@click.option(
    "--random-state", default=42, type=int, help="Random state", show_default=True
)
@click.option(
    "--n-splits",
    default=5,
    type=int,
    help="n splits for cross validation",
    show_default=True,
)
@click.option(
    "--use-scaler",
    default=True,
    type=bool,
    help="flag to use scaler for dataset",
    show_default=True,
)
@click.option(
    "--use-boruta",
    default=False,
    type=bool,
    help="flag to use boruta feature selection algorithm",
    show_default=True,
)
def train(
    dataset_path: Path,
    save_model_path: Path,
    model_name: str,
    random_state: int,
    n_splits: int,
    use_scaler: bool,
    use_boruta: bool,
):
    features, target = get_dataset(dataset_path)

    with mlflow.start_run():
        pipeline = create_pipeline(
            model_name=model_name,
            use_scaler=use_scaler,
            use_boruta=use_boruta,
            random_state=random_state,
        )

        accuracy = nestedCV(
            model_name,
            pipeline,
            features,
            target,
            scoring="accuracy",
            n_splits=n_splits,
        )
        micro_averaged_f1 = nestedCV(
            model_name,
            pipeline,
            features,
            target,
            scoring="f1_micro",
            n_splits=n_splits,
        )
        macro_averaged_f1 = nestedCV(
            model_name,
            pipeline,
            features,
            target,
            scoring="f1_macro",
            n_splits=n_splits,
        )
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("micro_averaged_f1", micro_averaged_f1)
        mlflow.log_metric("macro_averaged_f1", macro_averaged_f1)

        model = get_tuned_model(
            model_name, pipeline, features, target, n_splits=n_splits
        )
        mlflow.log_param("use_scaler", use_scaler)
        mlflow.log_param("use_boruta", use_boruta)
        mlflow.log_param("model_name", model_name)
        click.echo(f"Accuracy: {accuracy}.")
        click.echo(f"micro_averaged_f1: {micro_averaged_f1}.")
        click.echo(f"macro_averaged_f1: {macro_averaged_f1}.")

        dump(model, save_model_path)
        click.echo(f"Model is saved to {save_model_path}.")


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
