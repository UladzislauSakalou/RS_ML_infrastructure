from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier


def create_pipeline(random_state: int) -> Pipeline:
    scaler = StandardScaler()
    classifier = RandomForestClassifier(random_state=random_state)
    pipeline = Pipeline(steps=[("scaler", scaler), ("classifier", classifier)])
    return pipeline