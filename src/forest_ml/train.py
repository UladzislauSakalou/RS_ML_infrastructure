from pathlib import Path
import click
from .data import get_dataset
from .pipeline import create_pipeline
from .model import nestedCV, get_tuned_model, save_model
import mlflow
from typing import Any


@click.command()
@click.option(
    "-d",
    "--dataset-path",
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
    "--random-state", default=4, type=int, help="Random state", show_default=True
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
    use_scaler: bool,
    use_boruta: bool,
) -> None:
    features, target = get_dataset(dataset_path)

    with mlflow.start_run():
        pipeline = create_pipeline(use_scaler=use_scaler, use_boruta=use_boruta)
        features = pipeline.fit_transform(features)

        accuracy, precision_macro, macro_averaged_f1 = nestedCV(
            model_name, features, target, random_state, scoring="accuracy"
        )

        model, params = get_tuned_model(model_name, features, target, random_state)

        log_metrics(accuracy, precision_macro, macro_averaged_f1)
        log_params(params, use_scaler, use_boruta, model_name)

        click.echo(f"Accuracy: {accuracy}.")
        click.echo(f"micro_averaged_f1: {precision_macro}.")
        click.echo(f"macro_averaged_f1: {macro_averaged_f1}.")

        save_model(model, save_model_path)


def log_metrics(
    accuracy: float, precision_macro: float, macro_averaged_f1: float
) -> None:
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision_macro", precision_macro)
    mlflow.log_metric("macro_averaged_f1", macro_averaged_f1)


def log_params(
    params: dict[str, Any], use_scaler: bool, use_boruta: bool, model_name: str
) -> None:
    mlflow.log_params(params)
    mlflow.log_param("use_scaler", use_scaler)
    mlflow.log_param("use_boruta", use_boruta)
    mlflow.log_param("model_name", model_name)
