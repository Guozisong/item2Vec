import argparse
import json
import os
from pathlib import Path

import numpy as np
from tqdm import tqdm

from item2vec.io import load_item_index, load_index_item


def build_basket_indexes(baskets, item2index):
    basket_indexes = []
    for order in tqdm(baskets, desc='构建训练购物篮', unit='个', total=len(baskets)):
        sequence = [str(item2index[code]) for code in order if code in item2index]
        if 2 <= len(sequence) <= 20:
            basket_indexes.append(sequence)
    return basket_indexes


def prepare_order_baskets(dataframe, item2index, max_basket_size=30):
    required_columns = ('order_id', 'prod_id', 'dt')
    missing_columns = [column for column in required_columns if column not in dataframe.columns]
    if missing_columns:
        raise ValueError(f'Missing required columns: {", ".join(missing_columns)}')
    if dataframe.empty:
        raise ValueError('Input dataframe must not be empty')
    null_columns = [column for column in required_columns if dataframe[column].isna().any()]
    if null_columns:
        raise ValueError(f'Required fields must not contain nulls: {", ".join(null_columns)}')
    if (isinstance(max_basket_size, bool)
            or not isinstance(max_basket_size, (int, np.integer))
            or max_basket_size < 2):
        raise ValueError('max_basket_size must be at least 2')

    grouped = dataframe.groupby('order_id', sort=False)
    basket_indexes = []
    order_counts = np.zeros(len(item2index), dtype=np.int64)
    large_baskets = 0
    for _, order in tqdm(grouped, desc='构建训练购物篮', unit='单', total=grouped.ngroups):
        indexes = {
            int(item2index[product])
            for product in order['prod_id']
            if product in item2index
        }
        for index in indexes:
            order_counts[index] += 1
        if len(indexes) > max_basket_size:
            large_baskets += 1
        elif len(indexes) >= 2:
            basket_indexes.append([str(index) for index in sorted(indexes)])

    return basket_indexes, order_counts, {
        'orders': grouped.ngroups,
        'valid_baskets': len(basket_indexes),
        'large_baskets': large_baskets,
    }


def _validate_order_counts(order_counts, item_count):
    counts = np.asarray(order_counts)
    if (counts.shape != (item_count,)
            or not np.issubdtype(counts.dtype, np.integer)
            or np.any(counts < 0)):
        raise ValueError('order_counts must contain one nonnegative integer per catalog item')
    return counts


def train_item2vec(
    item2index,
    basket_indexes,
    order_counts,
    vector_size=128,
    max_basket_size=30,
    negative=15,
    epochs=10,
    min_order_count=5,
    workers=8,
):
    for name, value in [('vector_size', vector_size), ('max_basket_size', max_basket_size),
                        ('negative', negative), ('epochs', epochs),
                        ('min_order_count', min_order_count), ('workers', workers)]:
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f'{name} must be a positive integer')
    counts = _validate_order_counts(order_counts, len(item2index))
    eligible = {str(index) for index in np.flatnonzero(counts >= min_order_count)}
    training_baskets = []
    for basket in basket_indexes:
        filtered = [item for item in basket if item in eligible]
        if len(filtered) >= 2:
            training_baskets.append(filtered)
    if not training_baskets:
        raise RuntimeError('行为共现数据不足，未生成新的行为向量。')
    vectors = np.zeros((len(item2index), vector_size), dtype=np.float32)

    from gensim.models import Word2Vec

    model = Word2Vec(
        vector_size=vector_size,
        window=max_basket_size,
        shrink_windows=False,
        min_count=1,
        sg=1,
        negative=negative,
        sample=1e-4,
        workers=workers,
        epochs=epochs,
        alpha=0.002,
        min_alpha=0.0005,
    )
    model.build_vocab(training_baskets)
    print(f'开始训练行为向量：{epochs} 轮，维度 {vector_size}。', flush=True)
    model.train(
        training_baskets,
        total_examples=len(training_baskets),
        epochs=epochs,
        start_alpha=0.002,
        end_alpha=0.0005
    )
    # Only items with an eligible co-occurring partner have useful behavior context.
    supported = {item for basket in training_baskets for item in basket}
    for key in model.wv.index_to_key:
        if key in supported:
            vectors[int(key)] = model.wv[key]
    return vectors, model


def write_behavior_embedding(embedding, index2item, order_counts, metadata, downstream_dir):
    output_path = Path(downstream_dir) / 'behavior_item.npz'
    temporary_path = output_path.with_name('behavior_item.npz.tmp')
    item_ids = np.asarray([str(index2item[str(i)]) for i in range(len(index2item))])
    counts = _validate_order_counts(order_counts, len(index2item)).astype(np.int64)
    metadata_json = json.dumps(metadata, sort_keys=True)
    try:
        with temporary_path.open('wb') as stream:
            np.savez(stream, vectors=np.asarray(embedding, dtype=np.float32), item_ids=item_ids,
                     order_counts=counts, metadata=metadata_json)
        with np.load(temporary_path, allow_pickle=False) as artifact:
            required = {'vectors', 'item_ids', 'order_counts', 'metadata'}
            if not required.issubset(artifact.files):
                raise ValueError('Behavior artifact is missing required arrays')
            vectors = artifact['vectors']
            if vectors.ndim != 2:
                raise ValueError('Behavior vectors must be 2D')
            if vectors.shape[0] != len(item_ids):
                raise ValueError('Behavior vector row count must match the catalog')
            if vectors.shape[1] < 1:
                raise ValueError('Behavior vector dimension must be positive')
            if not np.isfinite(vectors).all():
                raise ValueError('Behavior vectors must be finite')
            if not np.any(vectors):
                raise ValueError('Behavior vector coverage must be nonzero')
            saved_counts = _validate_order_counts(artifact['order_counts'], len(item_ids))
            if saved_counts.dtype != np.int64:
                raise ValueError('order_counts must have int64 dtype')
            if not np.array_equal(artifact['item_ids'], item_ids):
                raise ValueError('Behavior item_ids must match the catalog exactly')
            try:
                saved_metadata = json.loads(artifact['metadata'].item())
            except (ValueError, TypeError) as error:
                raise ValueError('Behavior metadata must contain JSON') from error
            if saved_metadata != metadata:
                raise ValueError('Behavior metadata does not match training configuration')
        os.replace(temporary_path, output_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return output_path


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('raw_data_dir')
    parser.add_argument('downstream_dir')
    parser.add_argument('--vector-size', type=int, default=128)
    parser.add_argument('--max-basket-size', type=int, default=30)
    parser.add_argument('--min-order-count', type=int, default=5)
    parser.add_argument('--negative', type=int, default=15)
    parser.add_argument('--epochs', type=int, default=10)
    args = parser.parse_args(argv)

    import pandas as pd

    item2index = load_item_index(args.downstream_dir)
    index2item = load_index_item(args.downstream_dir)
    if (set(index2item) != {str(i) for i in range(len(index2item))}
            or len({str(item) for item in index2item.values()}) != len(index2item)
            or item2index != {str(item): int(i) for i, item in index2item.items()}):
        raise ValueError('商品索引不一致，请重新生成商品索引。')
    dataframe = pd.read_csv(
        os.path.join(args.raw_data_dir, 'order_item.csv'), dtype={'order_id': str, 'prod_id': str}
    )
    basket_indexes, order_counts, stats = prepare_order_baskets(
        dataframe, item2index, max_basket_size=args.max_basket_size
    )
    print(f'已读取 {len(dataframe)} 条行为，包含 {stats["orders"]} 个订单。')
    print(f'已构建 {stats["valid_baskets"]} 个有效购物篮，排除 {stats["large_baskets"]} 个超大购物篮，'
          f'商品目录共 {len(item2index)} 个商品。')
    metadata = {
        'vector_size': args.vector_size,
        'max_basket_size': args.max_basket_size,
        'negative': args.negative,
        'epochs': args.epochs,
        'min_order_count': args.min_order_count,
    }
    trained_embedding, _model = train_item2vec(
        item2index,
        basket_indexes,
        order_counts,
        **metadata,
    )
    output_path = write_behavior_embedding(
        trained_embedding, index2item, order_counts, metadata, args.downstream_dir
    )
    covered = np.count_nonzero(np.any(trained_embedding != 0, axis=1))
    print(f'行为向量覆盖 {covered} 个商品，其余 {len(item2index) - covered} 个商品回退到文本相似度。')
    print(f'行为向量已保存至：{output_path}')


if __name__ == '__main__':
    main()
