from pathlib import Path
import pandas as pd
import click
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from joblib import dump


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
def train(dataset_path: Path, save_model_path: Path):
    dataset = pd.read_csv(dataset_path)
    click.echo(f"Dataset shape: {dataset.shape}.")
    features = dataset.drop('Cover_Type', axis=1)
    target = dataset['Cover_Type']
    features_train, features_val, target_train, target_val = train_test_split(
        features, target, test_size=0.2, random_state=42
    )
    classifier = RandomForestClassifier(random_state=42).fit(features_train, target_train)
    accuracy = accuracy_score(target_val, classifier.predict(features_val))
    click.echo(f"Accuracy: {accuracy}.")
    dump(classifier, save_model_path)
    click.echo(f"Model is saved to {save_model_path}.")