import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


def write_json_atomic(value, path):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    try:
        with temporary.open('w', encoding='utf-8') as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_item_indexes(item_ids, output_path):
    item_ids = list(dict.fromkeys(item_ids))
    index2item = {index: item_id for index, item_id in enumerate(item_ids)}
    item2index = {item_id: index for index, item_id in enumerate(item_ids)}

    with open(os.path.join(output_path, "index2item.json"), "w", encoding="utf-8") as file:
        json.dump(index2item, file, ensure_ascii=False, indent=4)
    with open(os.path.join(output_path, "item2index.json"), "w", encoding="utf-8") as file:
        json.dump(item2index, file, ensure_ascii=False, indent=4)

    return item2index


def load_plm(plm_path):
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(plm_path)
    model = AutoModel.from_pretrained(plm_path)
    return tokenizer, model


def load_plm_embedding(path, file="item.feat1CLS"):
    feat_path = os.path.join(path, file)
    return np.fromfile(feat_path, dtype=np.float32).reshape(-1, 768)


def load_index_item(path):
    with open(os.path.join(path, "index2item.json"), "r", encoding="utf-8") as file:
        return json.load(file)


def load_item_index(path):
    with open(os.path.join(path, "item2index.json"), "r", encoding="utf-8") as file:
        return json.load(file)


def get_cosine_similarity(item_embedding, index2item, topk=10):
    norms = np.linalg.norm(item_embedding, axis=1, keepdims=True)
    normalized = np.divide(
        item_embedding,
        norms,
        out=np.zeros_like(item_embedding, dtype=np.float32),
        where=norms != 0,
    )
    sim_matrix = normalized @ normalized.T
    records = []

    for index in range(sim_matrix.shape[0]):
        item_id = index2item[str(index)]
        top_indices = np.argsort(-sim_matrix[index])
        top_indices = [other for other in top_indices if other != index][:topk]
        for other in top_indices:
            records.append((item_id, index2item[str(other)], float(sim_matrix[index, other])))

    return pd.DataFrame(records, columns=["master_prod_id", "slave_prod_id", "similarity"])
