from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from boruta import BorutaPy
from joblib import dump


def create_pipeline(use_scaler: bool, use_boruta: bool) -> Pipeline:
    pipeline_steps = []
    if use_scaler:
        pipeline_steps.append(
            (
                "scaler",
                ColumnTransformer(
                    transformers=[("scaler", StandardScaler(), get_num_columns())],
                    remainder="passthrough",
                ),
            )
        )

    if use_boruta:
        pipeline_steps.append(
            (
                "feature_selector",
                BorutaPy(
                    RandomForestClassifier(random_state=42, max_depth=13),
                    n_estimators="auto",
                    random_state=4,
                ),
            )
        )
    return Pipeline(steps=pipeline_steps)


def save_pipeline(pipeline: Pipeline, save_pipeline_path: Path):
    dump(pipeline, save_pipeline_path)


def get_num_columns() -> list[str]:
    return [
        "Elevation",
        "Aspect",
        "Slope",
        "Horizontal_Distance_To_Hydrology",
        "Vertical_Distance_To_Hydrology",
        "Horizontal_Distance_To_Roadways",
        "Hillshade_9am",
        "Hillshade_Noon",
        "Hillshade_3pm",
        "Horizontal_Distance_To_Fire_Points",
    ]
