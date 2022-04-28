from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier


def create_pipeline(
    use_scaler: bool,
    random_state: int,
    n_estimators: int,
    criterion: str,
    max_depth: int,
    min_samples_split: int,
    min_samples_leaf: int
) -> Pipeline:
    pipeline_steps = []
    if use_scaler:
        pipeline_steps.append(("scaler", StandardScaler()))
    pipeline_steps.append(("classifier", RandomForestClassifier(
        random_state=random_state,
        n_estimators=n_estimators,
        criterion=criterion,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf
    )))
    return Pipeline(steps=pipeline_steps)