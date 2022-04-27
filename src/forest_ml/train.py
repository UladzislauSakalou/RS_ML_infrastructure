from pathlib import Path
import pandas as pd
import click
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from joblib import dump
from .data import get_dataset


@click.command()
@click.option(
    '-d',
    '--dataset_path',
    default='data/train.csv',
    help='Path to the dataset',
    type=click.Path(exists=True, dir_okay=False, path_type=Path)
)
@click.option(
    "-s",
    "--save-model-path",
    default="data/model.joblib",
    type=click.Path(dir_okay=False, writable=True, path_type=Path)
)
@click.option(
    "--random-state",
    default=42,
    type=int
)
@click.option(
    "--test-split-ratio",
    default=0.2,
    type=click.FloatRange(0, 1, min_open=True, max_open=True),
)
def train(
    dataset_path: Path,
    save_model_path: Path,
    random_state: int,
    test_split_ratio: float
):
    features_train, features_val, target_train, target_val = get_dataset(
        dataset_path,
        random_state,
        test_split_ratio
    )
    classifier = RandomForestClassifier(random_state=random_state).fit(features_train, target_train)
    accuracy = accuracy_score(target_val, classifier.predict(features_val))
    click.echo(f"Accuracy: {accuracy}.")
    dump(classifier, save_model_path)
    click.echo(f"Model is saved to {save_model_path}.")