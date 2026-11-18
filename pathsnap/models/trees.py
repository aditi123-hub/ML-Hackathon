"""Compare one rule-based tree with a bagged ensemble of randomized trees."""

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


def decision_tree(seed, **parameters):
    settings = {"max_depth": 12, "min_samples_leaf": 30, "random_state": seed} | parameters
    return DecisionTreeClassifier(**settings)


def random_forest(seed, **parameters):
    settings = {
        "n_estimators": 64, "max_depth": 15, "min_samples_leaf": 10,
        "max_features": "sqrt", "n_jobs": 4, "random_state": seed,
    } | parameters
    return RandomForestClassifier(**settings)


def tree_space(trial):
    return {
        "max_depth": trial.suggest_int("max_depth", 6, 21),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 80, log=True),
    }


def forest_space(trial):
    return {
        "n_estimators": trial.suggest_int("n_estimators", 40, 100, step=20),
        "max_depth": trial.suggest_int("max_depth", 9, 23),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 3, 30, log=True),
        "max_features": trial.suggest_float("max_features", 0.4, 1.0),
    }
