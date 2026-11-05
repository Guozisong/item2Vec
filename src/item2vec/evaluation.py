"""Offline, time-separated evaluation of Item2Vec recommendations."""

import argparse
import json
from collections import Counter, defaultdict
from datetime import timedelta
from pathlib import Path

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


def prepare_truth(orders, allowed_items=None):
    """Prepare immutable-by-convention labels once, optionally for a warm cohort."""
    if not {'order_id', 'prod_id'}.issubset(orders.columns):
        raise ValueError('orders require order_id and prod_id')
    truth = defaultdict(Counter)
    for _, group in orders.groupby('order_id', sort=False):
        basket = list(dict.fromkeys(group.prod_id.astype(str)))
        if allowed_items is not None:
            basket = [item for item in basket if item in allowed_items]
            if len(basket) < 2:
                continue
        for source in basket:
            truth[source].update(target for target in basket if target != source)
    if not any(truth.values()):
        raise ValueError('test orders contain no multi-item baskets')
    return dict(truth)


def evaluate_predictions(orders, predictions, top_k=50, item_categories=None):
    return evaluate_rankings(prepare_truth(orders), predictions, top_k, item_categories)


def evaluate_rankings(truth, predictions, top_k=50, item_categories=None):
    if top_k < 1:
        raise ValueError('top_k must be positive')
    if not any(truth.values()):
        raise ValueError('test orders contain no multi-item baskets')
    if not {'master_prod_id', 'slave_prod_id'}.issubset(predictions.columns):
        raise ValueError('predictions require master_prod_id and slave_prod_id')
    recommendations = {}
    for source, group in predictions.groupby('master_prod_id', sort=False):
        recommendations[str(source)] = list(dict.fromkeys(group.slave_prod_id.astype(str)))[:top_k]
    unique_total = weighted_total = unique_hits = weighted_hits = 0
    category_total = {'same': 0, 'cross': 0}
    category_hits = {'same': 0, 'cross': 0}
    ndcgs = []
    recalls = []
    covered = 0
    for source, targets in truth.items():
        ranked = recommendations.get(source, [])
        covered += bool(ranked)
        unique_total += len(targets)
        weighted_total += sum(targets.values())
        hits = sum(target in targets for target in ranked)
        unique_hits += hits
        recalls.append(hits / len(targets) if targets else 0.)
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
        'macro_recall_at_k': float(np.mean(recalls)),
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


def load_validation_orders(processed_dir, validation_path):
    rows = pd.read_csv(validation_path, dtype=str, keep_default_na=False)
    required = {'order_id', 'prod_id', 'dt'}
    if not required.issubset(rows.columns) or rows.empty:
        raise ValueError('validation orders require nonempty order_id, prod_id, dt')
    if any(rows[column].str.strip().eq('').any() for column in required):
        raise ValueError('validation fields must not be blank')
    if not rows.dt.str.fullmatch(r'\d{8}').all():
        raise ValueError('validation dates must use YYYYMMDD')
    pd.to_datetime(rows.dt.unique(), format='%Y%m%d', errors='raise')
    validation_ids = set(rows.order_id)
    first_validation_date = rows.groupby('order_id', sort=False).dt.max().min()
    source = Path(processed_dir).parent / 'raw' / 'order_item.csv'
    if not source.is_file():
        raise ValueError('training raw orders are required to verify validation isolation')
    last_training_date = ''
    for chunk in pd.read_csv(source, usecols=['order_id', 'dt'], dtype=str,
                             keep_default_na=False, chunksize=250000):
        if chunk.order_id.isin(validation_ids).any():
            raise ValueError('training and validation order IDs overlap')
        last_training_date = max(last_training_date, chunk.dt.max())
    if not last_training_date or last_training_date >= first_validation_date:
        raise ValueError('training orders must strictly precede validation order dates')
    return rows


class BehaviorEvaluator:
    """Fixed labels/cohorts; all scoring is delegated to the production ranker."""

    def __init__(self, orders, index2item, train_counts, candidate_bias_weight=.4, item_categories=None,
                 behavior_score='legacy'):
        from item2vec.scoring import validate_candidate_bias_weight
        self.index2item = index2item
        self.counts = np.asarray(train_counts)
        if (self.counts.shape != (len(index2item),) or np.any(self.counts < 0)
                or not np.issubdtype(self.counts.dtype, np.integer)):
            raise ValueError('evaluation requires one training frequency per catalog item')
        self.bias_weight = validate_candidate_bias_weight(candidate_bias_weight)
        self.behavior_score = behavior_score
        self.categories = item_categories
        self.truth = prepare_truth(orders)
        active = {str(index2item[str(i)]) for i in np.flatnonzero(self.counts > 0)}
        self.warm = {source: Counter({target: count for target, count in targets.items() if target in active})
                     for source, targets in self.truth.items() if source in active}
        self.warm = {source: targets for source, targets in self.warm.items() if targets}
        if not self.warm:
            raise ValueError('validation has no evaluable warm multi-item baskets')
        low = {str(index2item[str(i)]) for i in np.flatnonzero((self.counts > 0) & (self.counts < 20))}
        self.low = {source: targets for source, targets in self.warm.items() if source in low}
        self.sources = [i for i in np.flatnonzero(self.counts > 0)
                        if str(index2item[str(i)]) in self.truth]

    def __call__(self, vectors, contexts, epoch=None):
        from item2vec.inference import rank_items
        from item2vec.scoring import compute_candidate_bias
        if not np.array_equal(np.any(vectors != 0, axis=1), self.counts > 0):
            raise ValueError('model active items differ from the fixed training cohort')
        top_k = min(50, len(self.index2item) - 1)
        bias = compute_candidate_bias(vectors, contexts, self.bias_weight) if self.behavior_score != 'cosine' else None
        predictions = rank_items(np.zeros((len(vectors), 1), dtype=np.float32), self.index2item,
                                 self.sources, top_k, behavior_vectors=vectors, text_weight=0.,
                                 behavior_order_counts=self.counts, behavior_context_vectors=contexts,
                                 recall_mode='complement', candidate_bias=bias, behavior_score=self.behavior_score)
        overall = evaluate_rankings(self.truth, predictions, top_k, self.categories)
        warm = evaluate_rankings(self.warm, predictions, top_k, self.categories)
        low = evaluate_rankings(self.low, predictions, top_k) if self.low else None
        return {'warm_macro_recall_at_50': warm['macro_recall_at_k'],
                'warm_ndcg_at_50': warm['mean_ndcg_at_k'], 'warm_sources': warm['sources'],
                'warm_source_coverage': warm['source_coverage'],
                'weighted_recall_at_50': overall['weighted_recall_at_k'],
                'pair_recall_at_50': overall['pair_recall_at_k'],
                'source_coverage': overall['source_coverage'], 'sources': overall['sources'],
                'low_frequency_macro_recall_at_50': low['macro_recall_at_k'] if low else None,
                'low_frequency_sources': low['sources'] if low else 0,
                'cross_category_weighted_recall_at_50': overall.get('cross_category_weighted_recall_at_k'),
                'distinct_candidates': overall['distinct_candidates'],
                'top100_candidate_share': overall['top100_candidate_share'],
                'active_items': int((self.counts > 0).sum())}


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
