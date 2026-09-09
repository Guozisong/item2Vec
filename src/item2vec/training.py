import argparse
import os
from pathlib import Path

import numpy as np
from tqdm import tqdm

from item2vec.io import load_item_index, load_plm_embedding


def build_basket_indexes(baskets, item2index):
    basket_indexes = []
    for order in tqdm(baskets, desc='构建训练购物篮', unit='个', total=len(baskets)):
        sequence = [str(item2index[code]) for code in order if code in item2index]
        if 2 <= len(sequence) <= 20:
            basket_indexes.append(sequence)
    return basket_indexes


def train_item2vec_with_bert_init(
    itemEmbedding,
    item2Index,
    baskets,
    lambda_bert=0.7,
    window=20,
    negative=15,
    epochs=10,
):
    from gensim.models import Word2Vec

    basket_index = build_basket_indexes(baskets, item2Index)
    model = Word2Vec(
        sentences=basket_index,
        vector_size=itemEmbedding.shape[1],
        window=window,
        min_count=5,
        sg=1,
        negative=negative,
        sample=1e-4,
        workers=8,
        epochs=1,
        alpha=0.0001
    )
    for i, key in enumerate(model.wv.index_to_key):
        index = int(key)
        model.wv.vectors[i] = lambda_bert * itemEmbedding[index] + (1 - lambda_bert) * model.wv.vectors[i]
    model.train(
        basket_index,
        total_examples=len(basket_index),
        epochs=epochs,
        start_alpha=0.002,
        end_alpha=0.0005
    )
    final_embedding = itemEmbedding.copy()
    for key in model.wv.index_to_key:
        final_embedding[int(key)] = model.wv[key]
    return final_embedding, model


def write_trained_embedding(embedding, downstream_dir):
    output_path = Path(downstream_dir) / "trained_item.featCLS"
    np.asarray(embedding, dtype=np.float32).tofile(output_path)
    return output_path


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('raw_data_dir')
    parser.add_argument('downstream_dir')
    parser.add_argument('--bert-weight', type=float, default=0.7)
    parser.add_argument('--window', type=int, default=20)
    parser.add_argument('--negative', type=int, default=15)
    parser.add_argument('--epochs', type=int, default=10)
    args = parser.parse_args(argv)

    import pandas as pd

    item_embedding = load_plm_embedding(args.downstream_dir)
    item2index = load_item_index(args.downstream_dir)
    dataframe = pd.read_csv(os.path.join(args.raw_data_dir, 'order_item.csv'))
    print(f'已读取 {len(dataframe)} 条行为，包含 {dataframe["user_id"].nunique()} 个用户。')
    baskets = (
        dataframe.groupby(['user_id', 'dt'])['prod_id']
        .apply(lambda values: list(dict.fromkeys(map(str, values))))
        .tolist()
    )
    print(f'已构建 {len(baskets)} 个有效购物篮，包含 {item_embedding.shape[0]} 个商品向量。')
    trained_embedding, _model = train_item2vec_with_bert_init(
        item_embedding,
        item2index,
        baskets,
        lambda_bert=args.bert_weight,
        window=args.window,
        negative=args.negative,
        epochs=args.epochs,
    )
    output_path = write_trained_embedding(trained_embedding, args.downstream_dir)
    print(f'训练向量已保存至：{output_path}')


if __name__ == '__main__':
    main()
