from email.contentmanager import raw_data_manager
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


def create_pipeline(
    model_name: str,
    use_scaler: bool,
    random_state: int,
    n_estimators: int,
    criterion: str,
    max_depth: int,
    min_samples_split: int,
    min_samples_leaf: int,
    logreg_c: float,
    penalty: str,
    max_iter: int
) -> Pipeline:
    pipeline_steps = []
    if use_scaler:
        pipeline_steps.append(("scaler", StandardScaler()))
    if model_name == 'rf':
        pipeline_steps.append(("classifier", RandomForestClassifier(
            random_state=random_state,
            n_estimators=n_estimators,
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf
        )))
    elif model_name == 'lr':
        pipeline_steps.append(('classifier', LogisticRegression(
            random_state=random_state,
            C=logreg_c,
            penalty=penalty,
            max_iter=max_iter
        )))
    return Pipeline(steps=pipeline_steps)
