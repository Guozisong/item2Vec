"""Candidate correction derived solely from trained Item2Vec vectors."""

import numpy as np


def validate_candidate_bias_weight(weight):
    if (isinstance(weight, (bool, np.bool_))
            or not isinstance(weight, (int, float, np.integer, np.floating))
            or not np.isfinite(weight) or weight < 0):
        raise ValueError('candidate_bias_weight must be finite and nonnegative')
    return float(weight)


def compute_candidate_bias(vectors, contexts, weight=.4):
    weight = validate_candidate_bias_weight(weight)
    vectors = np.asarray(vectors, dtype=np.float32)
    contexts = np.asarray(contexts, dtype=np.float32)
    if (vectors.ndim != 2 or contexts.shape != vectors.shape
            or not np.isfinite(vectors).all() or not np.isfinite(contexts).all()):
        raise ValueError('Behavior context vectors must match vectors and be finite')
    warm = np.flatnonzero(np.any(vectors != 0, axis=1) & np.any(contexts != 0, axis=1))
    bias = np.zeros(len(vectors), dtype=np.float32)
    if not len(warm):
        return bias
    sources = vectors[warm]
    neighbors = min(10, len(warm))
    for start in range(0, len(warm), 256):
        targets = warm[start:start + 256]
        scores = sources @ contexts[targets].T
        nearest = np.partition(scores, -neighbors, axis=0)[-neighbors:]
        bias[targets] = weight * nearest.mean(axis=0)
    return bias
