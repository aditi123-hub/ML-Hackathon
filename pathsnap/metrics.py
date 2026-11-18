"""Definitions shared by model selection, route evaluation, and API scoring."""

from dataclasses import dataclass
from time import perf_counter

import numpy as np


def predict_scores(model, features, valid):
    """Map positive-class probabilities back to the observation/candidate slots."""
    # Fit uses both boolean classes. sklearn sorts False before True, so the
    # second column is the probability of a match for every registered model.
    scores = np.full(valid.shape, -1.0, dtype=np.float64)
    scores[valid] = model.predict_proba(features[valid])[:, 1]
    return scores


def top_one(scores, ids):
    """Select the highest valid score at each observation and recover its arc ID."""
    return ids[np.arange(len(ids)), np.argmax(scores, axis=1)]


def direct_links(predictions, gaps, arcs, max_gap=20):
    """Same arc or a direct directed join among consecutive short-gap observations."""
    left, right = predictions[:-1], predictions[1:]
    eligible = (gaps[1:] > 0) & (gaps[1:] <= max_gap)
    joined = (left == right) | (arcs[left, 1] == arcs[right, 0])
    return int(np.count_nonzero(joined & eligible)), int(np.count_nonzero(eligible))


def divide(numerator, denominator):
    return float(numerator / denominator) if denominator else 0.0


def classification_metrics(confusion, loss):
    """Convert candidate-pair counts to binary and unweighted macro metrics."""
    tn, fp, fn, tp = confusion
    positive_f1 = divide(2 * tp, 2 * tp + fp + fn)
    negative_f1 = divide(2 * tn, 2 * tn + fp + fn)
    return {
        "accuracy": divide(tn + tp, sum(confusion)),
        "precision": divide(tp, tp + fp),
        "recall": divide(tp, tp + fn),
        "macro_f1": (positive_f1 + negative_f1) / 2,
        "log_loss": divide(loss, sum(confusion)),
        "confusion": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


@dataclass
class Totals:
    observations: int = 0
    correct: int = 0
    misses: int = 0
    links: int = 0
    steps: int = 0
    tn: int = 0
    fp: int = 0
    fn: int = 0
    tp: int = 0
    loss: float = 0
    inference_s: float = 0

    def add(self, model, batch):
        """Accumulate candidate-level and observation-level counts for one trip."""
        ids = batch.candidate_ids
        valid = ids >= 0
        started = perf_counter()  # Time model scoring separately from preparation.
        scores = predict_scores(model, batch.features, valid)
        self.inference_s += perf_counter() - started
        predictions = top_one(scores, ids)
        truth = ids == batch.reference_ids[:, None]
        target = truth[valid]
        # Clip only for log loss, where log(0) would be undefined.
        probabilities = np.clip(scores[valid], 1e-9, 1 - 1e-9)
        predicted = probabilities >= 0.5
        self.tn += int(np.count_nonzero(~target & ~predicted))
        self.fp += int(np.count_nonzero(~target & predicted))
        self.fn += int(np.count_nonzero(target & ~predicted))
        self.tp += int(np.count_nonzero(target & predicted))
        self.loss += float(np.sum(-np.log(np.where(target, probabilities, 1 - probabilities))))
        # A missing reference candidate never matches here. Still count the
        # observation in the denominator so misses cannot raise accuracy.
        self.correct += int(np.count_nonzero(predictions == batch.reference_ids))
        self.observations += len(predictions)
        self.misses += int(np.count_nonzero(~truth.any(axis=1)))
        links, steps = direct_links(predictions, batch.gaps, batch.arc_nodes)
        self.links += links
        self.steps += steps

    def summary(self):
        return classification_metrics((self.tn, self.fp, self.fn, self.tp), self.loss) | {
            "top1": divide(self.correct, self.observations),
            "candidate_miss_rate": divide(self.misses, self.observations),
            "direct_continuity": divide(self.links, self.steps),
            "observations": self.observations,
            "correct": self.correct,
            "misses": self.misses,
            "inference_s": self.inference_s,
        }


def evaluate(model, batches):
    totals = Totals()
    for batch in batches:
        totals.add(model, batch)
    return totals.summary()


def ordered_overlap(predicted, reference):
    """Sequence F1 using the exact length of the longest common subsequence.

    A bit set represents the columns where the LCS dynamic-programming row
    increases. Python's whole-integer subtraction propagates a match to the
    next eligible column, avoiding a Python loop over every column.
    """
    masks = {}
    for index, arc in enumerate(reference):
        masks[arc] = masks.get(arc, 0) | (1 << index)
    row = 0
    for arc in predicted:
        union = row | masks.get(arc, 0)
        row = union & ~(union - ((row << 1) | 1))
    return divide(2 * row.bit_count(), len(predicted) + len(reference))
