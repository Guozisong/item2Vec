"""Build a reproducible Item2Vec basket dataset from order lines."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from item2vec.io import load_index_item, load_item_index


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _item_ids(embeddings_dir):
    index2item = load_index_item(embeddings_dir)
    item2index = load_item_index(embeddings_dir)
    if (set(index2item) != {str(i) for i in range(len(index2item))}
            or item2index != {str(item): i for i, item in
                              enumerate(index2item[str(j)] for j in range(len(index2item)))}):
        raise ValueError('商品索引不一致')
    return np.asarray([str(index2item[str(i)]) for i in range(len(index2item))]), item2index


def preprocess_baskets(raw_dir, embeddings_dir, processed_dir, max_basket_size=30,
                       min_co_basket_orders=5):
    if max_basket_size < 2 or min_co_basket_orders < 1:
        raise ValueError('max_basket_size must be >= 2 and min_co_basket_orders >= 1')
    raw_dir = Path(raw_dir)
    embeddings_dir = Path(embeddings_dir)
    processed_dir = Path(processed_dir)
    item_ids, item2index = _item_ids(embeddings_dir)
    source = raw_dir / 'order_item.csv'
    rows = pd.read_csv(source, dtype=str, keep_default_na=False)
    required = {'order_id', 'prod_id', 'dt'}
    if not required.issubset(rows.columns):
        raise ValueError('order_item.csv requires order_id, prod_id, and dt')
    if rows.empty:
        raise ValueError('order_item.csv must not be empty')
    for column in required:
        if rows[column].str.strip().eq('').any():
            raise ValueError(f'{column} must not be blank')
    dates = pd.to_datetime(rows.dt, format='%Y%m%d', errors='coerce')
    if dates.isna().any() or not rows.dt.str.fullmatch(r'\d{8}').all():
        raise ValueError('dt must use a valid YYYYMMDD date')

    counts = dict(input_rows=len(rows), orders=0, duplicate_product_rows=0,
                  unknown_product_rows=0, cross_date_orders=0,
                  zero_known_baskets=0, single_known_baskets=0,
                  initial_valid_baskets=0, large_baskets=0,
                  dropped_after_support=0, final_baskets=0)
    all_order_counts = np.zeros(len(item_ids), dtype=np.int64)
    valid_basket_counts = np.zeros(len(item_ids), dtype=np.int64)
    initial = []
    for order_id, group in rows.groupby('order_id', sort=True):
        counts['orders'] += 1
        products = group.prod_id.tolist()
        counts['duplicate_product_rows'] += len(products) - len(set(products))
        counts['unknown_product_rows'] += sum(product not in item2index for product in products)
        if group.dt.nunique() > 1:
            counts['cross_date_orders'] += 1
        indexes = sorted({int(item2index[product]) for product in products
                          if product in item2index})
        all_order_counts[indexes] += 1
        size = len(indexes)
        if size == 0:
            counts['zero_known_baskets'] += 1
        elif size == 1:
            counts['single_known_baskets'] += 1
        elif size > max_basket_size:
            counts['large_baskets'] += 1
        else:
            counts['initial_valid_baskets'] += 1
            valid_basket_counts[indexes] += 1
            initial.append((str(order_id), group.dt.max(), indexes))

    processed_dir.mkdir(parents=True, exist_ok=True)
    baskets_path = processed_dir / 'baskets.jsonl'
    support_path = processed_dir / 'support.npz'
    manifest_path = processed_dir / 'manifest.json'
    baskets_temp = processed_dir / 'baskets.jsonl.tmp'
    support_temp = processed_dir / 'support.npz.tmp'
    manifest_temp = processed_dir / 'manifest.json.tmp'
    try:
        with baskets_temp.open('w', encoding='utf-8') as stream:
            for order_id, date, indexes in initial:
                kept = [index for index in indexes
                        if valid_basket_counts[index] >= min_co_basket_orders]
                if len(kept) < 2:
                    counts['dropped_after_support'] += 1
                    continue
                record = {'order_id': order_id, 'dt': date, 'items': kept}
                stream.write(json.dumps(record, ensure_ascii=False, separators=(',', ':')) + '\n')
                counts['final_baskets'] += 1
        if not counts['final_baskets']:
            raise ValueError('no qualifying baskets remain after preprocessing')
        with support_temp.open('wb') as stream:
            np.savez(stream, item_ids=item_ids, all_order_counts=all_order_counts,
                     valid_basket_counts=valid_basket_counts)
        manifest = {
            'version': 1,
            'parameters': {'max_basket_size': max_basket_size,
                           'min_co_basket_orders': min_co_basket_orders},
            'inputs': {'orders_sha256': _sha256(source),
                       'index_sha256': _sha256(embeddings_dir / 'index2item.json')},
            'outputs': {'baskets_sha256': _sha256(baskets_temp),
                        'support_sha256': _sha256(support_temp)},
            'counts': counts,
        }
        with manifest_temp.open('w', encoding='utf-8') as stream:
            json.dump(manifest, stream, ensure_ascii=False, indent=2)
        os.replace(baskets_temp, baskets_path)
        os.replace(support_temp, support_path)
        os.replace(manifest_temp, manifest_path)
    finally:
        for path in (baskets_temp, support_temp, manifest_temp):
            path.unlink(missing_ok=True)
    return counts


def load_processed_baskets(processed_dir, embeddings_dir, max_basket_size=None,
                           min_co_basket_orders=None):
    processed_dir = Path(processed_dir)
    embeddings_dir = Path(embeddings_dir)
    with (processed_dir / 'manifest.json').open(encoding='utf-8') as stream:
        manifest = json.load(stream)
    if manifest.get('version') != 1:
        raise ValueError('unsupported basket manifest version')
    parameters = manifest['parameters']
    if max_basket_size is not None and parameters['max_basket_size'] != max_basket_size:
        raise ValueError('basket size differs from preprocessing manifest')
    if (min_co_basket_orders is not None
            and parameters['min_co_basket_orders'] != min_co_basket_orders):
        raise ValueError('minimum basket support differs from preprocessing manifest')
    if _sha256(embeddings_dir / 'index2item.json') != manifest['inputs']['index_sha256']:
        raise ValueError('processed baskets do not match item index')
    source = processed_dir.parent / 'raw' / 'order_item.csv'
    if source.exists() and _sha256(source) != manifest['inputs']['orders_sha256']:
        raise ValueError('processed baskets do not match source orders')
    baskets_path = processed_dir / 'baskets.jsonl'
    support_path = processed_dir / 'support.npz'
    if (_sha256(baskets_path) != manifest['outputs']['baskets_sha256']
            or _sha256(support_path) != manifest['outputs']['support_sha256']):
        raise ValueError('processed basket files do not match manifest')
    item_ids, _ = _item_ids(embeddings_dir)
    with np.load(support_path, allow_pickle=False) as support:
        if not np.array_equal(support['item_ids'], item_ids):
            raise ValueError('basket support does not match item index')
        all_counts = support['all_order_counts']
        valid_counts = support['valid_basket_counts']
    if (all_counts.shape != item_ids.shape or valid_counts.shape != item_ids.shape
            or np.any(all_counts < 0) or np.any(valid_counts < 0)):
        raise ValueError('basket support counts are invalid')
    baskets = []
    with baskets_path.open(encoding='utf-8') as stream:
        for line in stream:
            record = json.loads(line)
            indexes = record['items']
            if (not 2 <= len(indexes) <= parameters['max_basket_size']
                    or indexes != sorted(set(indexes))
                    or any(not 0 <= index < len(item_ids)
                           or valid_counts[index] < parameters['min_co_basket_orders']
                           for index in indexes)):
                raise ValueError('processed basket contains invalid items')
            baskets.append([str(index) for index in indexes])
    if len(baskets) != manifest['counts']['final_baskets']:
        raise ValueError('processed basket count does not match manifest')
    return baskets, all_counts, valid_counts, manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description='Prepare Item2Vec training baskets')
    parser.add_argument('raw_dir')
    parser.add_argument('embeddings_dir')
    parser.add_argument('processed_dir')
    parser.add_argument('--max-basket-size', type=int, default=30)
    parser.add_argument('--min-co-basket-orders', type=int, default=5)
    args = parser.parse_args(argv)
    counts = preprocess_baskets(args.raw_dir, args.embeddings_dir, args.processed_dir,
                                args.max_basket_size, args.min_co_basket_orders)
    print(f'购物篮处理完成：{counts["orders"]} 单，'
          f'{counts["final_baskets"]} 个有效训练篮；报告位于 '
          f'{Path(args.processed_dir) / "manifest.json"}')


if __name__ == '__main__':
    main()
