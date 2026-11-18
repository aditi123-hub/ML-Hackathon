"""A small neural classifier. Fixed epochs keep the validation split trip-wise."""

from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def shallow_mlp(seed, **parameters):
    settings = {
        "hidden_layer_sizes": (32, 16), "alpha": 0.001, "batch_size": 512,
        "max_iter": 38, "learning_rate_init": 0.002, "early_stopping": False,
        "random_state": seed,
    } | parameters
    return make_pipeline(StandardScaler(), MLPClassifier(**settings))


def search_space(trial):
    return {
        "alpha": trial.suggest_float("alpha", 0.0001, 0.05, log=True),
        "learning_rate_init": trial.suggest_float("learning_rate_init", 0.0003, 0.006, log=True),
        "max_iter": trial.suggest_int("max_iter", 24, 60, step=12),
    }
