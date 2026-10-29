"""Content-selected anchors for Item2Vec source vectors."""

from pathlib import Path

import numpy as np
import pandas as pd


def load_categories(embeddings_dir):
    path = Path(embeddings_dir).parent / 'raw' / 'item.csv'
    if not path.is_file():
        return {}
    items = pd.read_csv(path, dtype=str)
    if not {'prod_id', 'prod_description'}.issubset(items.columns):
        raise ValueError('item.csv requires prod_id and prod_description')
    items = items.drop_duplicates('prod_id')
    return {str(row.prod_id): row.prod_description.lstrip('{').split('，商品品牌为', 1)[0]
            for row in items.itertuples(index=False)
            if isinstance(row.prod_description, str) and '，商品品牌为' in row.prod_description}


def transfer_behavior(source, text, behavior, mapping, available, categories,
                      min_similarity=.8, anchor_indexes=None):
    category = categories.get(str(mapping[str(source)]))
    if category is None or not np.any(text[source]):
        return None, []
    pool = np.flatnonzero(available) if anchor_indexes is None else anchor_indexes
    anchors = np.asarray([i for i in pool
                          if i != source and categories.get(str(mapping[str(i)])) == category],
                         dtype=int)
    if len(anchors) < 2:
        return None, []
    similarities = text[anchors] @ text[source]
    keep = similarities >= min_similarity
    anchors, similarities = anchors[keep], similarities[keep]
    if len(anchors) < 2:
        return None, []
    order = np.lexsort((anchors, -similarities))[:5]
    anchors, similarities = anchors[order], similarities[order]
    weights = np.exp((similarities - similarities.max()) / .1)
    weights /= weights.sum()
    vector = weights @ behavior[anchors]
    return (vector, anchors.tolist()) if np.any(vector) else (None, [])
