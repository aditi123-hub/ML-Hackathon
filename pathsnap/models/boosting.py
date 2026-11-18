"""Histogram gradient boosting learns interactions with shallow additive trees."""

from sklearn.ensemble import HistGradientBoostingClassifier


def histogram_boosting(seed, **parameters):
    # Histogram bins keep repeated split searches practical at this sample size.
    # The learning rate, leaf budget, leaf support and L2 term constrain the
    # additive corrections to binary log loss.
    settings = {
        "max_iter": 140, "max_leaf_nodes": 31, "learning_rate": 0.12,
        "min_samples_leaf": 25, "l2_regularization": 0.3,
        "early_stopping": False, "random_state": seed,
    } | parameters
    # A fixed iteration count avoids observation-level internal validation.
    return HistGradientBoostingClassifier(**settings)


def search_space(trial):
    # Tune iteration count jointly with learning rate. Leaf limits, support,
    # and L2 control interactions and shrink noisy candidate scores.
    return {
        "max_iter": trial.suggest_int("max_iter", 80, 220, step=20),
        "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 15, 63),
        "learning_rate": trial.suggest_float("learning_rate", 0.04, 0.2, log=True),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 12, 70),
        "l2_regularization": trial.suggest_float("l2_regularization", 0.01, 5, log=True),
    }
