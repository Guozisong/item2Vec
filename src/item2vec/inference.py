import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from item2vec.io import load_index_item, load_plm_embedding


COLUMNS = ["master_prod_id", "slave_prod_id", "similarity"]
MODE_TEXT_WEIGHTS = {'similar': .85, 'complement': .20, 'hybrid': .60}


def resolve_text_weight(recall_mode, text_weight):
    if not isinstance(recall_mode, str) or recall_mode not in MODE_TEXT_WEIGHTS:
        raise ValueError('recall_mode must be similar, complement, or hybrid')
    if text_weight is None:
        return MODE_TEXT_WEIGHTS[recall_mode]
    if (isinstance(text_weight, (bool, np.bool_))
            or not isinstance(text_weight, (int, float, np.integer, np.floating))
            or not np.isfinite(text_weight) or not 0 <= text_weight <= 1):
        raise ValueError('text_weight must be finite and between 0 and 1')
    return text_weight


def _validate_behavior_order_counts(order_counts, item_count):
    counts = np.asarray(order_counts)
    if (counts.shape != (item_count,)
            or not np.issubdtype(counts.dtype, np.integer)
            or np.any(counts < 0)):
        raise ValueError('behavior_order_counts must contain one nonnegative integer per item')
    return counts


def load_trained_artifacts(downstream_dir):
    vectors = load_plm_embedding(downstream_dir, file="item.feat1CLS")
    index2item = load_index_item(downstream_dir)
    validate_artifacts(vectors, index2item)
    path = Path(downstream_dir) / 'behavior_item.npz'
    if not path.is_file():
        raise FileNotFoundError(f'{path} 不存在，请运行 bash scripts/train.sh 生成独立行为向量。')
    with np.load(path, allow_pickle=False) as artifact:
        behavior_vectors = artifact['vectors']
        item_ids = artifact['item_ids']
        try:
            order_counts = _validate_behavior_order_counts(artifact['order_counts'], len(index2item))
        except (KeyError, ValueError) as error:
            raise ValueError('行为向量 order_counts 缺失或无效，请重新运行 bash scripts/train.sh。') from error
    validate_artifacts(behavior_vectors, index2item)
    expected_ids = np.asarray([str(index2item[str(i)]) for i in range(len(index2item))])
    if not np.array_equal(item_ids, expected_ids):
        raise ValueError('行为向量与商品索引不匹配，请重新运行 bash scripts/train.sh。')
    return vectors, index2item, behavior_vectors, order_counts


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
    if vectors.shape[1] == 0:
        raise ValueError('vectors must have at least one dimension')
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


def rank_items(
    vectors,
    index2item,
    source_indexes,
    top_k,
    block_size=512,
    show_progress=False,
    behavior_vectors=None,
    text_weight=0.7,
    behavior_order_counts=None,
    full_confidence_orders=50,
):
    vectors = np.asarray(vectors)
    validate_artifacts(vectors, index2item)
    validate_top_k(top_k, vectors.shape[0])
    if isinstance(block_size, bool) or not isinstance(block_size, (int, np.integer)) or block_size < 1:
        raise ValueError("block_size must be a positive integer")
    text_weight = resolve_text_weight('hybrid', text_weight)
    if (isinstance(full_confidence_orders, (bool, np.bool_))
            or not isinstance(full_confidence_orders, (int, np.integer))
            or full_confidence_orders < 1):
        raise ValueError('full_confidence_orders must be a positive integer')
    if behavior_order_counts is not None or behavior_vectors is not None:
        behavior_order_counts = _validate_behavior_order_counts(behavior_order_counts, len(index2item))

    normalized = _normalize_vectors(vectors)
    behavior_normalized = None
    if behavior_vectors is not None:
        behavior_vectors = np.asarray(behavior_vectors)
        validate_artifacts(behavior_vectors, index2item)
        behavior_normalized = _normalize_vectors(behavior_vectors)
        available = np.any(behavior_vectors != 0, axis=1)
        confidence = np.minimum(behavior_order_counts / full_confidence_orders, 1.)
    records = []
    source_indexes = list(source_indexes)

    block_starts = range(0, len(source_indexes), block_size)
    if show_progress:
        block_starts = tqdm(
            block_starts,
            desc="计算商品相似度",
            unit="块",
            total=(len(source_indexes) + block_size - 1) // block_size,
        )

    for block_start in block_starts:
        block_sources = source_indexes[block_start:block_start + block_size]
        similarities = normalized[block_sources] @ normalized.T
        if behavior_normalized is not None and text_weight < 1:
            behavior_scores = behavior_normalized[block_sources] @ behavior_normalized.T
            pair_available = available[block_sources, None] & available[None, :]
            pair_confidence = np.minimum(confidence[block_sources, None], confidence[None, :])
            effective_behavior_weight = (1 - text_weight) * pair_confidence * pair_available
            similarities = ((1 - effective_behavior_weight) * similarities
                            + effective_behavior_weight * behavior_scores)
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


def query_item(downstream_dir, item_id, top_k=10, text_weight=None,
               recall_mode='hybrid', full_confidence_orders=50):
    text_weight = resolve_text_weight(recall_mode, text_weight)
    print("正在加载训练向量与索引…")
    vectors, index2item, behavior_vectors, order_counts = load_trained_artifacts(downstream_dir)
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

    print(f"正在查询商品 {requested_id} 的 Top-{top_k} 相似商品…")
    result = rank_items(vectors, index2item, [int(source_index)], top_k,
                        behavior_vectors=behavior_vectors, text_weight=text_weight,
                        behavior_order_counts=order_counts, full_confidence_orders=full_confidence_orders)
    output_path = Path(downstream_dir) / f"query_{safe_item_filename(item_id)}.csv"
    result.to_csv(output_path, index=False)
    print(f"查询完成，共写入 {len(result)} 条结果：{output_path}")
    return output_path


def export_all(downstream_dir, top_k=10, block_size=512, text_weight=None,
               recall_mode='hybrid', full_confidence_orders=50):
    text_weight = resolve_text_weight(recall_mode, text_weight)
    print("正在加载训练向量与索引…")
    vectors, index2item, behavior_vectors, order_counts = load_trained_artifacts(downstream_dir)
    print("正在计算全量商品相似度…")
    result = rank_items(
        vectors,
        index2item,
        range(vectors.shape[0]),
        top_k,
        block_size=block_size,
        show_progress=True,
        behavior_vectors=behavior_vectors,
        text_weight=text_weight,
        behavior_order_counts=order_counts,
        full_confidence_orders=full_confidence_orders,
    )
    output_path = Path(downstream_dir) / "item_cosine_similarity.csv"
    result.to_csv(output_path, index=False)
    print(f"导出完成，共写入 {len(result)} 条结果：{output_path}")
    return output_path


def _build_parser():
    parser = argparse.ArgumentParser(description="Query trained Item2Vec similarities")
    subparsers = parser.add_subparsers(dest="command", required=True)

    query_parser = subparsers.add_parser("query")
    query_parser.add_argument("downstream_dir")
    query_parser.add_argument("item_id")
    query_parser.add_argument("--top-k", type=int, default=10)
    query_parser.add_argument('--recall-mode', choices=MODE_TEXT_WEIGHTS, default='hybrid')
    query_parser.add_argument('--text-weight', type=float, default=None)
    query_parser.add_argument('--full-confidence-orders', type=int, default=50)

    export_parser = subparsers.add_parser("export")
    export_parser.add_argument("downstream_dir")
    export_parser.add_argument("--top-k", type=int, default=10)
    export_parser.add_argument("--block-size", type=int, default=512)
    export_parser.add_argument('--recall-mode', choices=MODE_TEXT_WEIGHTS, default='hybrid')
    export_parser.add_argument('--text-weight', type=float, default=None)
    export_parser.add_argument('--full-confidence-orders', type=int, default=50)
    return parser


def main(argv=None):
    args = _build_parser().parse_args(argv)
    if args.command == "query":
        query_item(args.downstream_dir, args.item_id, top_k=args.top_k, text_weight=args.text_weight,
                   recall_mode=args.recall_mode, full_confidence_orders=args.full_confidence_orders)
    else:
        export_all(args.downstream_dir, top_k=args.top_k, block_size=args.block_size,
                   text_weight=args.text_weight, recall_mode=args.recall_mode,
                   full_confidence_orders=args.full_confidence_orders)


if __name__ == "__main__":
    main()
