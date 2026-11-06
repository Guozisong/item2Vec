import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
from item2vec.io import load_item_index, load_index_item, write_json_atomic
from item2vec.preprocessing import load_processed_baskets, _sha256
from item2vec.sampling import validate_integer


def _validate_order_counts(order_counts, item_count):
    counts = np.asarray(order_counts)
    if (counts.shape != (item_count,)
            or not np.issubdtype(counts.dtype, np.integer)
            or np.any(counts < 0)):
        raise ValueError('order_counts must contain one nonnegative integer per catalog item')
    return counts


def write_behavior_embedding(embedding, index2item, order_counts, metadata, downstream_dir):
    metadata = dict(metadata)
    output_path = Path(downstream_dir) / 'behavior_item.npz'
    temporary_path = output_path.with_name('behavior_item.npz.tmp')
    item_ids = np.asarray([str(index2item[str(i)]) for i in range(len(index2item))])
    counts = _validate_order_counts(order_counts, len(index2item)).astype(np.int64)
    metadata_json = json.dumps(metadata, sort_keys=True)
    try:
        with tempfile.TemporaryFile() as stream:
            arrays = {'vectors': np.asarray(embedding, dtype=np.float32),
                      'item_ids': item_ids, 'order_counts': counts, 'metadata': metadata_json}
            np.savez(stream, **arrays)
            stream.seek(0)
            with np.load(stream, allow_pickle=False) as artifact:
                required = {'vectors', 'item_ids', 'order_counts', 'metadata'}
                if not required.issubset(artifact.files):
                    raise ValueError('Behavior artifact is missing required arrays')
                vectors = artifact['vectors']
                if vectors.ndim != 2:
                    raise ValueError('Behavior vectors must be 2D')
                if vectors.shape[0] != len(item_ids):
                    raise ValueError('Behavior vector row count must match the catalog')
                if vectors.shape[1] < 1:
                    raise ValueError('Behavior vector dimension must be positive')
                if not np.isfinite(vectors).all():
                    raise ValueError('Behavior vectors must be finite')
                if not np.any(vectors):
                    raise ValueError('Behavior vector coverage must be nonzero')
                saved_counts = _validate_order_counts(artifact['order_counts'], len(item_ids))
                if saved_counts.dtype != np.int64:
                    raise ValueError('order_counts must have int64 dtype')
                if not np.array_equal(artifact['item_ids'], item_ids):
                    raise ValueError('Behavior item_ids must match the catalog exactly')
                try:
                    saved_metadata = json.loads(artifact['metadata'].item())
                except (ValueError, TypeError) as error:
                    raise ValueError('Behavior metadata must contain JSON') from error
                if saved_metadata != metadata:
                    raise ValueError('Behavior metadata does not match training configuration')
            stream.seek(0)
            with temporary_path.open('wb') as output:
                shutil.copyfileobj(stream, output)
        os.replace(temporary_path, output_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return output_path


class _Once(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        seen = getattr(namespace, '_supplied_training_options', set())
        if self.dest in seen:
            parser.error(f'{option_string} may only be specified once')
        setattr(namespace, self.dest, values)
        namespace._supplied_training_options = seen | {self.dest}


def main(argv=None):
    from item2vec.attention import DEFAULTS, validate_config, train_attention, save_checkpoint
    from item2vec.evaluation import BehaviorEvaluator, load_validation_orders, load_item_categories

    parser = argparse.ArgumentParser(description='Train attention Item2Vec from filtered baskets.')
    parser.add_argument('processed_dir')
    parser.add_argument('downstream_dir')
    parser.add_argument('--vector-size', type=int, default=128)
    parser.add_argument('--max-basket-size', type=int, default=30)
    parser.add_argument('--min-order-count', type=int, default=5)
    parser.add_argument('--negative', type=int, default=15)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--check-options', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--trainer', choices=['attention'], default='attention', action=_Once)
    parser.add_argument('--seed', type=int, action=_Once)
    for option, kind in [('validation-orders', str), ('patience', int), ('min-delta', float),
                         ('device', str), ('attention-heads', int), ('temperature', float),
                         ('pair-loss-weight', float), ('context-batch-size', int), ('learning-rate', float)]:
        parser.add_argument('--' + option, type=kind, action=_Once)
    args = parser.parse_args(argv)
    validate_integer(args.min_order_count, 'min_order_count')
    validate_integer(args.max_basket_size, 'max_basket_size', 2)
    supplied = getattr(args, '_supplied_training_options', set())
    if not args.validation_orders and ({'patience', 'min_delta'} & supplied):
        raise ValueError('patience and min_delta require --validation-orders')
    trainer_config = validate_config({name: getattr(args, name) for name in DEFAULTS
                                     if hasattr(args, name) and getattr(args, name) is not None})
    if args.check_options:
        return

    item2index = load_item_index(args.downstream_dir)
    index2item = load_index_item(args.downstream_dir)
    if (set(index2item) != {str(i) for i in range(len(index2item))}
            or len({str(item) for item in index2item.values()}) != len(index2item)
            or item2index != {str(item): int(i) for i, item in index2item.items()}):
        raise ValueError('商品索引不一致，请重新生成商品索引。')
    baskets, all_order_counts, _, manifest = load_processed_baskets(
        args.processed_dir, args.downstream_dir,
        max_basket_size=args.max_basket_size,
        min_co_basket_orders=args.min_order_count, compact=True,
    )
    stats = manifest['counts']
    print(f'已读取 {stats["input_rows"]} 条行为，包含 {stats["orders"]} 个订单。')
    print(f'已构建 {stats["final_baskets"]} 个有效购物篮，排除 {stats["large_baskets"]} 个超大购物篮，'
          f'商品目录共 {len(item2index)} 个商品。')
    print('训练模型：attention；召回打分：共享商品向量余弦。', flush=True)
    metadata = {
        'max_basket_size': args.max_basket_size,
        'min_order_count': args.min_order_count,
        'min_basket_size': manifest['parameters'].get('min_basket_size', 2),
        'training_inputs': manifest['inputs'],
    }
    evaluator = None
    if args.validation_orders:
        rows = load_validation_orders(args.processed_dir, args.validation_orders)
        item_csv = Path(args.processed_dir).parent / 'raw' / 'item.csv'
        categories = load_item_categories(item_csv) if item_csv.is_file() else None
        counts = np.bincount(baskets.items, minlength=len(item2index))
        evaluator = BehaviorEvaluator(rows, index2item, counts, item_categories=categories)
        metadata['validation_sha256'] = _sha256(args.validation_orders)
    destination = Path(args.downstream_dir)
    destination.mkdir(parents=True, exist_ok=True)
    history = []

    def on_epoch(record):
        history.append(record)
        write_json_atomic(history, destination.parent / 'history.json')
        print(json.dumps(record, ensure_ascii=False, allow_nan=False), flush=True)

    def on_best(vectors, details):
        write_behavior_embedding(vectors, index2item, all_order_counts,
                                 {**metadata, **details}, destination)

    result = train_attention(baskets, len(item2index), trainer_config,
                             validate_epoch=evaluator, on_epoch=on_epoch, on_best=on_best)
    metadata.update(result['metadata'])
    save_checkpoint(result, destination / 'attention_model.pt')
    if evaluator:
        write_json_atomic(result['history'][metadata['best_epoch']-1]['metrics'],
                          destination.parent / 'validation.json')
    output_path = write_behavior_embedding(result['vectors'], index2item, all_order_counts,
                                           metadata, destination)
    covered = np.count_nonzero(np.any(result['vectors'] != 0, axis=1))
    print(f'行为向量覆盖 {covered} 个商品，其余 {len(item2index) - covered} 个商品'
          '在 similar/hybrid 中回退到文本相似度；complement 模式尝试行为迁移后回退文本。')
    print(f'行为向量已保存至：{output_path}')


if __name__ == '__main__':
    main()
