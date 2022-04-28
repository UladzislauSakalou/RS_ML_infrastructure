from pathlib import Path
import click
from joblib import dump
from .data import get_dataset
from .pipeline import create_pipeline
from sklearn.model_selection import KFold, cross_val_score
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
    "--n-splits",
    default=5,
    type=int,
    help='n splits for cross validation',
    show_default=True
)
@click.option(
    "--use-scaler",
    default=True,
    type=bool,
    help='flag to use scaler for dataset',
    show_default=True
)
@click.option(
    "--n-estimators",
    default=100,
    type=int,
    help='The number of trees in the Random forest.',
    show_default=True
)
@click.option(
    "--criterion",
    default='gini',
    type=click.Choice(['gini', 'entropy']),
    help='The function to measure the quality of a split. Supported criteria are “gini” for the Gini impurity and “entropy” for the information gain.',
    show_default=True
)
@click.option(
    "--max-depth",
    default=None,
    type=int,
    help='The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.',
    show_default=True
)
@click.option(
    "--min-samples-split",
    default=2,
    type=int,
    help='The minimum number of samples required to split an internal node.',
    show_default=True
)
@click.option(
    "--min-samples-leaf",
    default=1,
    type=int,
    help='The minimum number of samples required to be at a leaf node.',
    show_default=True
)
def train(
    dataset_path: Path,
    save_model_path: Path,
    random_state: int,
    n_splits: int,
    use_scaler: bool,
    n_estimators: int,
    criterion: str,
    max_depth: int,
    min_samples_split: int,
    min_samples_leaf: int
):
    features, target = get_dataset(dataset_path)
    
    with mlflow.start_run():
        pipeline = create_pipeline(
            use_scaler=use_scaler,
            random_state=random_state,
            n_estimators=n_estimators,
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf
        )
        k_fold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        accuracy = cross_val_score(pipeline, features, target, cv=k_fold, scoring='accuracy').mean()
        micro_averaged_f1 = cross_val_score(pipeline, features, target, cv=k_fold, scoring='f1_micro').mean()
        macro_averaged_f1 = cross_val_score(pipeline, features, target, cv=k_fold, scoring='f1_macro').mean()
        mlflow.log_param("use_scaler", use_scaler)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("micro_averaged_f1", micro_averaged_f1)
        mlflow.log_metric("macro_averaged_f1", macro_averaged_f1)
        click.echo(f"Accuracy: {accuracy}.")
        click.echo(f"micro_averaged_f1: {micro_averaged_f1}.")
        click.echo(f"macro_averaged_f1: {macro_averaged_f1}.")
        pipeline.fit(features, target)
        dump(pipeline, save_model_path)
        click.echo(f"Model is saved to {save_model_path}.")