import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from item2vec.io import load_index_item, load_plm_embedding


COLUMNS = ["master_prod_id", "slave_prod_id", "similarity"]


def load_trained_artifacts(downstream_dir):
    vectors = load_plm_embedding(downstream_dir, file="trained_item.featCLS")
    index2item = load_index_item(downstream_dir)
    validate_artifacts(vectors, index2item)
    return vectors, index2item


def safe_item_filename(item_id):
    filename = re.sub(r"[^A-Za-z0-9._-]", "_", str(item_id))
    if not filename or not filename.strip("."):
        return "item"
    return filename


def validate_top_k(top_k, item_count):
    if isinstance(top_k, (bool, np.bool_)) or not isinstance(top_k, (int, np.integer)):
        raise ValueError("top_k must be an integer")
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
    if not np.isfinite(vectors).all():
        raise ValueError("vectors must contain only finite values")


def _normalize_vectors(vectors):
    if not np.issubdtype(vectors.dtype, np.floating):
        vectors = vectors.astype(np.float64)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return np.divide(
        vectors,
        norms,
        out=np.zeros_like(vectors),
        where=norms != 0,
    )


def rank_items(vectors, index2item, source_indexes, top_k, block_size=512):
    vectors = np.asarray(vectors)
    validate_artifacts(vectors, index2item)
    validate_top_k(top_k, vectors.shape[0])
    if block_size < 1:
        raise ValueError("block_size must be at least 1")

    normalized = _normalize_vectors(vectors)
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


def query_item(downstream_dir, item_id, top_k=10):
    vectors, index2item = load_trained_artifacts(downstream_dir)
    requested_id = str(item_id)
    source_index = next(
        (
            index
            for index, mapped_item in index2item.items()
            if str(mapped_item) == requested_id
        ),
        None,
    )
    if source_index is None:
        raise ValueError(f"Unknown item ID: {requested_id}")

    result = rank_items(vectors, index2item, [int(source_index)], top_k)
    output_path = Path(downstream_dir) / f"query_{safe_item_filename(item_id)}.csv"
    result.to_csv(output_path, index=False)
    return output_path


def export_all(downstream_dir, top_k=10, block_size=512):
    vectors, index2item = load_trained_artifacts(downstream_dir)
    result = rank_items(
        vectors,
        index2item,
        range(vectors.shape[0]),
        top_k,
        block_size=block_size,
    )
    output_path = Path(downstream_dir) / "item_cosine_similarity.csv"
    result.to_csv(output_path, index=False)
    return output_path


def _build_parser():
    parser = argparse.ArgumentParser(description="Query trained Item2Vec similarities")
    subparsers = parser.add_subparsers(dest="command", required=True)

    query_parser = subparsers.add_parser("query")
    query_parser.add_argument("downstream_dir")
    query_parser.add_argument("item_id")
    query_parser.add_argument("--top-k", type=int, default=10)

    export_parser = subparsers.add_parser("export")
    export_parser.add_argument("downstream_dir")
    export_parser.add_argument("--top-k", type=int, default=10)
    export_parser.add_argument("--block-size", type=int, default=512)
    return parser


def main(argv=None):
    args = _build_parser().parse_args(argv)
    if args.command == "query":
        query_item(args.downstream_dir, args.item_id, top_k=args.top_k)
    else:
        export_all(args.downstream_dir, top_k=args.top_k, block_size=args.block_size)


if __name__ == "__main__":
    main()
