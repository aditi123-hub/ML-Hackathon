"""The five probability-producing baselines share one experiment protocol."""

from . import boosting, linear, neural, trees


FACTORIES = {
    "logistic": linear.logistic,
    "tree": trees.decision_tree,
    "forest": trees.random_forest,
    "boosting": boosting.histogram_boosting,
    "mlp": neural.shallow_mlp,
}
SEARCH_SPACES = {
    "logistic": linear.search_space,
    "tree": trees.tree_space,
    "forest": trees.forest_space,
    "boosting": boosting.search_space,
    "mlp": neural.search_space,
}
