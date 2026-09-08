import numpy as np
import pandas as pd


COLUMNS = ["master_prod_id", "slave_prod_id", "similarity"]


def validate_top_k(top_k, item_count):
    if not 1 <= top_k < item_count:
        raise ValueError("top_k must satisfy 1 <= top_k < item_count")


def validate_artifacts(vectors, index2item):
    if vectors.ndim != 2:
        raise ValueError("vectors must be a 2D array")
    if vectors.shape[0] != len(index2item):
        raise ValueError("vector row count must equal mapping length")
    expected_keys = {str(index) for index in range(len(index2item))}
    if set(index2item) != expected_keys:
        raise ValueError("mapping keys must be contiguous string indexes")


def rank_items(vectors, index2item, source_indexes, top_k, block_size=512):
    vectors = np.asarray(vectors)
    validate_artifacts(vectors, index2item)
    validate_top_k(top_k, vectors.shape[0])
    if block_size < 1:
        raise ValueError("block_size must be at least 1")

    float_vectors = vectors.astype(np.float64, copy=False)
    norms = np.linalg.norm(float_vectors, axis=1, keepdims=True)
    normalized = np.divide(
        float_vectors,
        norms,
        out=np.zeros_like(float_vectors),
        where=norms != 0,
    )
    records = []
    source_indexes = list(source_indexes)

    for block_start in range(0, len(source_indexes), block_size):
        block_sources = source_indexes[block_start:block_start + block_size]
        similarities = normalized[block_sources] @ normalized.T
        for block_index, scores in enumerate(similarities):
            source_index = block_sources[block_index]
            ranked_indexes = np.argsort(-scores, kind="stable")
            ranked_indexes = ranked_indexes[ranked_indexes != source_index][:top_k]
            source_item = index2item[str(source_index)]
            records.extend(
                (
                    source_item,
                    index2item[str(target_index)],
                    float(scores[target_index]),
                )
                for target_index in ranked_indexes
            )

    return pd.DataFrame(records, columns=COLUMNS)
