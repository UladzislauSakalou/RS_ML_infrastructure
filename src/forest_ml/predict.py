from pathlib import Path
import click
from joblib import load
import pandas as pd
import numpy as np
from os import path


@click.command()
@click.option(
    "-d",
    "--model-path",
    default="data/model.joblib",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to the model file",
    show_default=True,
)
@click.option(
    "-d",
    "--test-dataset-path",
    default="data/test.csv",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to the dataset",
    show_default=True,
)
@click.option(
    "-d",
    "--sample-submission-path",
    default="data/sampleSubmission.csv",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to the sample submission",
    show_default=True,
)
@click.option(
    "-d",
    "--pipeline-path",
    default="data/pipeline.joblib",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to the pipeline",
    show_default=True,
)
@click.option(
    "-s",
    "--save-pred-path",
    default="data/submission.csv",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    help="Path to the predictions to save",
    show_default=True,
)
def predict(
    model_path: Path,
    test_dataset_path: Path,
    sample_submission_path: Path,
    pipeline_path: Path,
    save_pred_path,
) -> None:
    model = load(model_path)
    test_data = pd.read_csv(test_dataset_path, index_col="Id")
    if path.exists(pipeline_path):
        pipeline = load(pipeline_path)
        test_data = pipeline.transform(np.array(test_data))
    preds = model.predict(test_data)
    submission = pd.read_csv(sample_submission_path)
    submission["Cover_Type"] = preds
    submission[["Id", "Cover_Type"]].to_csv(save_pred_path, index=False)
