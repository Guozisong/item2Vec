"""Offline, time-separated evaluation of Item2Vec recommendations."""

import argparse
import json
from collections import Counter, defaultdict
from datetime import timedelta

import numpy as np
import pandas as pd


def split_orders(rows, validation_start, test_start):
    if not {'order_id', 'prod_id', 'dt'}.issubset(rows.columns):
        raise ValueError('orders require order_id, prod_id, and dt')
    if validation_start >= test_start:
        raise ValueError('validation_start must precede test_start')
    dates = rows.groupby('order_id', sort=False).dt.max()
    period = dates.map(lambda dt: 0 if dt < validation_start else
                       1 if dt < test_start else 2)
    labels = rows.order_id.map(period)
    return tuple(rows.loc[labels == number].copy() for number in range(3))


def load_item_categories(item_csv):
    items = pd.read_csv(item_csv, dtype=str)
    if not {'prod_id', 'prod_description'}.issubset(items.columns):
        raise ValueError('item CSV requires prod_id and prod_description')
    descriptions = items.drop_duplicates('prod_id').set_index('prod_id').prod_description
    return {str(item): description.lstrip('{').split('，商品品牌为', 1)[0]
            for item, description in descriptions.items()
            if isinstance(description, str) and '，商品品牌为' in description}


def evaluate_predictions(orders, predictions, top_k=50, item_categories=None):
    if top_k < 1:
        raise ValueError('top_k must be positive')
    if not {'order_id', 'prod_id'}.issubset(orders.columns):
        raise ValueError('orders require order_id and prod_id')
    if not {'master_prod_id', 'slave_prod_id'}.issubset(predictions.columns):
        raise ValueError('predictions require master_prod_id and slave_prod_id')
    truth = defaultdict(Counter)
    for _, group in orders.groupby('order_id', sort=False):
        basket = list(dict.fromkeys(group.prod_id.astype(str)))
        for source in basket:
            truth[source].update(target for target in basket if target != source)
    if not truth:
        raise ValueError('test orders contain no multi-item baskets')
    recommendations = {}
    for source, group in predictions.groupby('master_prod_id', sort=False):
        recommendations[str(source)] = list(dict.fromkeys(group.slave_prod_id.astype(str)))[:top_k]
    unique_total = weighted_total = unique_hits = weighted_hits = 0
    category_total = {'same': 0, 'cross': 0}
    category_hits = {'same': 0, 'cross': 0}
    ndcgs = []
    covered = 0
    for source, targets in truth.items():
        ranked = recommendations.get(source, [])
        covered += bool(ranked)
        unique_total += len(targets)
        weighted_total += sum(targets.values())
        unique_hits += sum(target in targets for target in ranked)
        weighted_hits += sum(targets[target] for target in ranked)
        if item_categories is not None and source in item_categories:
            for target, count in targets.items():
                if target not in item_categories:
                    continue
                group = 'same' if item_categories[source] == item_categories[target] else 'cross'
                category_total[group] += count
                if target in ranked:
                    category_hits[group] += count
        dcg = sum(targets[target] / np.log2(rank + 2)
                  for rank, target in enumerate(ranked))
        ideal = sum(count / np.log2(rank + 2)
                    for rank, count in enumerate(sorted(targets.values(), reverse=True)[:top_k]))
        ndcgs.append(dcg / ideal if ideal else 0.)
    candidate_counts = Counter(target for source in truth
                               for target in recommendations.get(source, []))
    candidate_total = sum(candidate_counts.values())
    metrics = {
        'sources': len(truth),
        'pair_recall_at_k': unique_hits / unique_total,
        'weighted_recall_at_k': weighted_hits / weighted_total,
        'mean_ndcg_at_k': float(np.mean(ndcgs)),
        'source_coverage': covered / len(truth),
        'distinct_candidates': len(candidate_counts),
        'top100_candidate_share': (sum(count for _, count in candidate_counts.most_common(100))
                                   / candidate_total if candidate_total else 0.),
    }
    if item_categories is not None:
        for group in ('same', 'cross'):
            metrics[f'{group}_category_weighted_recall_at_k'] = (
                category_hits[group] / category_total[group] if category_total[group] else None
            )
            metrics[f'{group}_category_pair_count'] = category_total[group]
    return metrics


def main(argv=None):
    parser = argparse.ArgumentParser(description='Time split and evaluate Item2Vec exports')
    commands = parser.add_subparsers(dest='command', required=True)
    split = commands.add_parser('split')
    split.add_argument('orders')
    split.add_argument('--train', required=True)
    split.add_argument('--validation', required=True)
    split.add_argument('--test', required=True)
    split.add_argument('--validation-days', type=int, default=14)
    split.add_argument('--test-days', type=int, default=14)
    evaluate = commands.add_parser('evaluate')
    evaluate.add_argument('test_orders')
    evaluate.add_argument('recommendations', nargs='+')
    evaluate.add_argument('--top-k', type=int, default=50)
    evaluate.add_argument('--item-csv')
    args = parser.parse_args(argv)
    if args.command == 'split':
        if args.validation_days < 1 or args.test_days < 1:
            parser.error('holdout days must be positive')
        rows = pd.read_csv(args.orders, dtype=str)
        last_day = pd.to_datetime(rows.dt.max(), format='%Y%m%d')
        test_start = last_day - timedelta(days=args.test_days - 1)
        validation_start = test_start - timedelta(days=args.validation_days)
        parts = split_orders(rows, validation_start.strftime('%Y%m%d'),
                             test_start.strftime('%Y%m%d'))
        if any(part.empty for part in parts):
            parser.error('split produced an empty period; shorten the holdout windows')
        for part, path in zip(parts, (args.train, args.validation, args.test)):
            part.to_csv(path, index=False)
            print(f'{path}: {part.order_id.nunique()} orders, {len(part)} rows')
    else:
        orders = pd.read_csv(args.test_orders, dtype=str)
        categories = load_item_categories(args.item_csv) if args.item_csv else None
        for path in args.recommendations:
            results = pd.read_csv(path, dtype={'master_prod_id': str, 'slave_prod_id': str})
            print(json.dumps({'file': path, **evaluate_predictions(orders, results, args.top_k,
                                                                 categories)},
                             ensure_ascii=False))


if __name__ == '__main__':
    main()
