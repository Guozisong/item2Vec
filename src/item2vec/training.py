import argparse
import os
from collections import Counter
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


def train_item2vec(
    item2Index,
    baskets,
    vector_size=128,
    window=20,
    negative=15,
    epochs=10,
    basket_index=None,
):
    for name, value in [('vector_size', vector_size), ('window', window),
                        ('negative', negative), ('epochs', epochs)]:
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f'{name} must be a positive integer')
    if basket_index is None:
        basket_index = build_basket_indexes(baskets, item2Index)
    vectors = np.zeros((len(item2Index), vector_size), dtype=np.float32)
    counts = Counter(item for basket in basket_index for item in basket)
    eligible = {item for item, count in counts.items() if count >= 5}
    if not any(len(set(basket) & eligible) >= 2 for basket in basket_index):
        print('行为共现数据不足，所有商品将回退到文本相似度。')
        return vectors, None

    from gensim.models import Word2Vec

    model = Word2Vec(
        vector_size=vector_size,
        window=window,
        min_count=5,
        sg=1,
        negative=negative,
        sample=1e-4,
        workers=8,
        epochs=epochs,
        alpha=0.002,
        min_alpha=0.0005,
    )
    model.build_vocab(basket_index)
    print(f'开始训练行为向量：{epochs} 轮，维度 {vector_size}。', flush=True)
    model.train(
        basket_index,
        total_examples=len(basket_index),
        epochs=epochs,
        start_alpha=0.002,
        end_alpha=0.0005
    )
    # Only items with an eligible co-occurring partner have useful behavior context.
    supported = set()
    for basket in basket_index:
        members = set(basket) & eligible
        if len(members) >= 2:
            supported.update(members)
    for key in model.wv.index_to_key:
        if key in supported:
            vectors[int(key)] = model.wv[key]
    return vectors, model


def write_behavior_embedding(embedding, index2item, downstream_dir):
    output_path = Path(downstream_dir) / 'behavior_item.npz'
    item_ids = np.asarray([str(index2item[str(i)]) for i in range(len(index2item))])
    np.savez(output_path, vectors=np.asarray(embedding, dtype=np.float32), item_ids=item_ids)
    return output_path


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('raw_data_dir')
    parser.add_argument('downstream_dir')
    parser.add_argument('--vector-size', type=int, default=128)
    parser.add_argument('--window', type=int, default=20)
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
    dataframe = pd.read_csv(os.path.join(args.raw_data_dir, 'order_item.csv'), dtype={'prod_id': str})
    print(f'已读取 {len(dataframe)} 条行为，包含 {dataframe["user_id"].nunique()} 个用户。')
    baskets = (
        dataframe.groupby(['user_id', 'dt'])['prod_id']
        .apply(lambda values: list(dict.fromkeys(map(str, values))))
        .tolist()
    )
    basket_index = build_basket_indexes(baskets, item2index)
    print(f'已构建 {len(basket_index)} 个有效购物篮，商品目录共 {len(item2index)} 个商品。')
    trained_embedding, _model = train_item2vec(
        item2index,
        baskets,
        vector_size=args.vector_size,
        window=args.window,
        negative=args.negative,
        epochs=args.epochs,
        basket_index=basket_index,
    )
    output_path = write_behavior_embedding(trained_embedding, index2item, args.downstream_dir)
    covered = np.count_nonzero(np.any(trained_embedding != 0, axis=1))
    print(f'行为向量覆盖 {covered} 个商品，其余 {len(item2index) - covered} 个商品回退到文本相似度。')
    print(f'行为向量已保存至：{output_path}')


if __name__ == '__main__':
    main()
