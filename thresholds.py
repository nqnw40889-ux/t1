"""Counts and rates for a fixed set of labeled predictions."""
import numpy as np


def confusion_counts(truth, scores, threshold):
    truth = np.asarray(truth, dtype=bool)
    predicted = np.asarray(scores) >= threshold
    return {
        "tp": int((truth & predicted).sum()),
        "fp": int((~truth & predicted).sum()),
        "fn": int((truth & ~predicted).sum()),
        "tn": int((~truth & ~predicted).sum()),
    }


def rates(counts):
    tp, fp, fn = (counts[key] for key in ("tp", "fp", "fn"))
    return {
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
    }
