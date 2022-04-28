from pathlib import Path
import click
from sklearn.metrics import accuracy_score
from joblib import dump
from .data import get_dataset
from .pipeline import create_pipeline
import mlflow


@click.command()
@click.option(
    '-d',
    '--dataset_path',
    default='data/train.csv',
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help='Path to the dataset',
    show_default=True
)
@click.option(
    "-s",
    "--save-model-path",
    default="data/model.joblib",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    help='Path to the model to save',
    show_default=True
)
@click.option(
    "--random-state",
    default=42,
    type=int,
    help='Random state',
    show_default=True
)
@click.option(
    "--test-split-ratio",
    default=0.2,
    type=click.FloatRange(0, 1, min_open=True, max_open=True),
    help='Test split ratio',
    show_default=True
)
@click.option(
    "--use-scaler",
    default=True,
    type=bool,
    help='flag to use scaler for dataset',
    show_default=True
)
def train(
    dataset_path: Path,
    save_model_path: Path,
    random_state: int,
    test_split_ratio: float,
    use_scaler: bool
):
    features_train, features_val, target_train, target_val = get_dataset(
        dataset_path,
        random_state,
        test_split_ratio
    )
    with mlflow.start_run():
        pipeline = create_pipeline(use_scaler=use_scaler, random_state=random_state).fit(features_train, target_train)
        accuracy = accuracy_score(target_val, pipeline.predict(features_val))
        mlflow.log_param("use_scaler", use_scaler)
        mlflow.log_metric("accuracy", accuracy)
        click.echo(f"Accuracy: {accuracy}.")
        dump(pipeline, save_model_path)
        click.echo(f"Model is saved to {save_model_path}.")