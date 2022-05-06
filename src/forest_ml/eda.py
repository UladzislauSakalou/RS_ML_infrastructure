from pandas_profiling import ProfileReport
from .data import get_dataset
import pandas as pd
import click
from pathlib import Path


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
    "--save-report-path",
    default="data/eda.html",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    help="Path to the report to save",
    show_default=True,
)
def eda(dataset_path: Path, save_report_path: Path) -> None:
    features, target = get_dataset(dataset_path)
    profile = ProfileReport(
        pd.concat([features, target], axis=1), title="Pandas Profiling Report"
    )
    profile.to_file(save_report_path)
