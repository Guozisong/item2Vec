import argparse
import os
from pathlib import Path

import numpy as np

from item2vec.io import load_item_index, load_plm_embedding


def build_basket_indexes(baskets, item2index):
    basket_indexes = []
    for order in baskets:
        sequence = [str(item2index[code]) for code in order if code in item2index]
        if 2 <= len(sequence) <= 20:
            basket_indexes.append(sequence)
    return basket_indexes


def train_item2vec_with_bert_init(itemEmbedding, item2Index, baskets, lambda_bert=0.7):
    from gensim.models import Word2Vec

    basket_index = build_basket_indexes(baskets, item2Index)
    model = Word2Vec(
        sentences=basket_index,
        vector_size=itemEmbedding.shape[1],
        window=20,
        min_count=5,
        sg=1,
        negative=15,
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
        epochs=10,
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('raw_data_dir')
    parser.add_argument('downstream_dir')
    args = parser.parse_args()

    import pandas as pd

    item_embedding = load_plm_embedding(args.downstream_dir)
    print(type(item_embedding))
    print(item_embedding.shape)
    print(item_embedding[0, :])
    item2index = load_item_index(args.downstream_dir)
    dataframe = pd.read_csv(os.path.join(args.raw_data_dir, 'order_item.csv'))
    baskets = (
        dataframe.groupby(['user_id', 'dt'])['prod_id']
        .apply(lambda values: list(dict.fromkeys(map(str, values))))
        .tolist()
    )
    print(baskets)
    trained_embedding, _model = train_item2vec_with_bert_init(item_embedding, item2index, baskets)
    write_trained_embedding(trained_embedding, args.downstream_dir)


if __name__ == '__main__':
    main()
