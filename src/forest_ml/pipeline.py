from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from boruta import BorutaPy


def create_pipeline(
    model_name: str, use_scaler: bool, use_boruta: bool, random_state: int
) -> Pipeline:
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
                    random_state=42,
                ),
            )
        )
    if model_name == "rf":
        pipeline_steps.append(
            ("classifier", RandomForestClassifier(random_state=random_state))
        )
    elif model_name == "lr":
        pipeline_steps.append(
            ("classifier", LogisticRegression(random_state=random_state))
        )
    return Pipeline(steps=pipeline_steps)


def get_num_columns():
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
