import os.path
import pandas as pd
import numpy as np
import json
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModel, AutoTokenizer



def itemIndex(csv_file_path, output_path):
    prod_id = list(dict.fromkeys(pd.read_csv(csv_file_path)['prod_id']))
    index2item = {idx: pid for idx, pid in enumerate(prod_id)}
    item2index = {pid: idx for idx, pid in enumerate(prod_id)}

    with open(os.path.join(output_path, 'index2item.json'), 'w', encoding='utf-8') as f:
        json.dump(index2item, f, ensure_ascii=False, indent=4)

    with open(os.path.join(output_path, 'item2index.json'), 'w', encoding='utf-8') as f:
        json.dump(item2index, f, ensure_ascii=False, indent=4)
    print("itemlen:", len(prod_id))


def load_plm(plm_path):
    tokenizer = AutoTokenizer.from_pretrained(plm_path)
    model = AutoModel.from_pretrained(plm_path)
    return tokenizer, model


def load_plm_embedding(path, file="lianhua_item.feat1CLS"):
    feat_path = os.path.join(path, file)
    loaded_feat = np.fromfile(feat_path, dtype=np.float32).reshape(-1, 768)
    return loaded_feat


def load_index_item(path):
    with open(os.path.join(path, 'index2item.json'), "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def load_item_index(path):
    with open(os.path.join(path, 'item2index.json'), "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def get_cosine_similarity(itemEmbedding, index2Item, topk=10):
    sim_matrix = cosine_similarity(itemEmbedding)
    records = []

    for i in range(sim_matrix.shape[0]):
        item_id = index2Item[str(i)]
        top_indices = np.argsort(-sim_matrix[i])
        top_indices = [idx for idx in top_indices if idx != i][:topk]

        for j in top_indices:
            sim_item_id = index2Item[str(j)]
            similarity = float(sim_matrix[i, j])
            records.append((item_id, sim_item_id, similarity))

    # 转为DataFrame
    df = pd.DataFrame(records, columns=["master_prod_id", "slave_prod_id", "similarity"])
    return df



if __name__ == '__main__':
    itemIndex("dataset/raw/lianhua.csv", './dataset/downstream/')
