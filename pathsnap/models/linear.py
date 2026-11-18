"""Linear baseline. Scaling is fitted inside each model's training fold."""

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def logistic(seed, **parameters):
    settings = {"C": 1.0, "max_iter": 200, "random_state": seed} | parameters
    return make_pipeline(StandardScaler(), LogisticRegression(**settings))


def search_space(trial):
    return {"C": trial.suggest_float("C", 0.01, 10, log=True)}
