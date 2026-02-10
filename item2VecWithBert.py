import numpy as np
import pandas as pd
import os
from gensim.models import Word2Vec
from utils import load_item_index, load_plm_embedding, load_index_item, get_cosine_similarity

def train_item2vec_with_bert_init(
    itemEmbedding,
    item2Index,
    baskets,
    lambda_bert=0.7   # 融合系数（核心参数）
):
    """
    lambda_bert:
        1.0  -> 纯 BERT
        0.0  -> 纯 Item2Vec
    """

    # =============== Step 1: basket -> index ===============
    basket_index = []
    for order in baskets:
        seq = []
        for code in order:
            if code in item2Index:
                seq.append(str(item2Index[code]))
        if len(seq) >= 2 and len(seq) <= 20 :
            basket_index.append(seq)

    # =============== Step 2: 构建 Word2Vec（只建结构） ===============
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

    # =============== Step 3: BERT + 随机向量融合初始化 ===============
    for i, key in enumerate(model.wv.index_to_key):
        idx = int(key)
        bert_vec = itemEmbedding[idx]
        w2v_vec = model.wv.vectors[i]

        # 核心：融合
        model.wv.vectors[i] = (
            lambda_bert * bert_vec +
            (1 - lambda_bert) * w2v_vec
        )

    # =============== Step 4: 行为微调（控制学习率） ===============
    model.train(
        basket_index,
        total_examples=len(basket_index),
        epochs=10,
        start_alpha=0.002,
        end_alpha=0.0005
    )

    # =============== Step 5: 写回全量 embedding ===============
    final_embedding = itemEmbedding.copy()
    for i, key in enumerate(model.wv.index_to_key):
        final_embedding[int(key)] = model.wv[key]

    return final_embedding, model


if __name__ == '__main__':
    raw_path = './dataset/raw/'
    feat_path = './dataset/downstream/'
    # 获取商品bert编码
    itemEmbedding = load_plm_embedding(feat_path)
    print(type(itemEmbedding))
    print(itemEmbedding.shape)
    print(itemEmbedding[0, :])
    # 获取商品id和商品索引对应关系
    item2Index = load_item_index(feat_path)
    index2Item = load_index_item(feat_path)
    # 构建购物车
    df = pd.read_csv(os.path.join(raw_path, 'order_item.csv'))

    baskets = (
        df.groupby(['user_id', 'dt'])['prod_id']
            .apply(lambda x: list(dict.fromkeys(map(str, x))))
            .tolist()
    )
    print(baskets)
    trained_embedding, model = train_item2vec_with_bert_init(itemEmbedding, item2Index, baskets)
    trained_embedding.astype(np.float32).tofile(os.path.join(feat_path, "trained_item.featCLS"))
    item_similarity = get_cosine_similarity(itemEmbedding, index2Item, topk=10)
    item_similarity.to_csv("./item_cosine_similarity.csv", index=False, encoding="utf-8-sig")
