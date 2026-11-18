"""Linear baseline. Scaling is fitted inside each model's training fold."""

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def logistic(seed, **parameters):
    # Standardize metres, seconds, fractions, and cosines within the fold so
    # L2 regularization does not chiefly reflect the original units.
    # The 200-step solver budget lets the log-loss fit converge on these features.
    settings = {"C": 1.0, "max_iter": 200, "random_state": seed} | parameters
    return make_pipeline(StandardScaler(), LogisticRegression(**settings))


def search_space(trial):
    # C is inverse regularization. Explore orders of magnitude around the
    # baseline because equal additive steps are less informative near zero.
    return {"C": trial.suggest_float("C", 0.01, 10, log=True)}
