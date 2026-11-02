"""Weighted Item2Vec SGNS with current-basket negative exclusion."""

import hashlib
import math
import resource
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from item2vec.sampling import BasketTable, NegativeSampler, basket_weights, iter_pair_batches, validate_integer


DEFAULTS = dict(vector_size=128, negative=15, epochs=10, basket_pair_budget=0,
                negative_exclusion='basket', hard_negative_ratio=0., hard_negative_warmup=2,
                hard_negative_pool_size=128, pair_batch_size=4096, seed=1,
                device='auto', patience=3, min_delta=.0001)


def validate_config(config):
    unknown = set(config) - DEFAULTS.keys()
    if unknown:
        raise ValueError(f'unknown SGNS options: {sorted(unknown)}')
    config = {**DEFAULTS, **config}
    for name in ('vector_size', 'negative', 'epochs', 'hard_negative_pool_size', 'pair_batch_size', 'patience'):
        validate_integer(config[name], name)
    for name in ('seed', 'basket_pair_budget', 'hard_negative_warmup'):
        validate_integer(config[name], name, 0)
    for name in ('hard_negative_ratio', 'min_delta'):
        value = config[name]
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float))
                or not math.isfinite(value) or value < 0):
            raise ValueError(f'{name} must be finite and nonnegative')
    if config['hard_negative_ratio'] >= 1:
        raise ValueError('hard_negative_ratio must be less than 1')
    if int(config['negative'] * config['hard_negative_ratio']) > config['hard_negative_pool_size']:
        raise ValueError('hard_negative_pool_size is smaller than the hard sample count')
    if config['negative_exclusion'] not in ('target', 'basket'):
        raise ValueError('negative_exclusion must be target or basket')
    if config['device'] not in ('auto', 'cpu', 'cuda'):
        raise ValueError('device must be auto, cpu or cuda')
    if config['device'] == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA was requested but is unavailable')
    return config


class SGNSModel(nn.Module):
    def __init__(self, item_count, dimension):
        super().__init__()
        self.inputs = nn.Embedding(item_count, dimension, sparse=True)
        self.contexts = nn.Embedding(item_count, dimension, sparse=True)
        nn.init.uniform_(self.inputs.weight, -.5/dimension, .5/dimension)
        nn.init.zeros_(self.contexts.weight)

    def score_candidates(self, sources, candidates):
        return (self.inputs(sources)[:, None, :] * self.contexts(candidates)).sum(dim=-1)

    def forward(self, sources, targets, negatives):
        source_vectors = self.inputs(sources)
        positive = (source_vectors * self.contexts(targets)).sum(dim=-1)
        negative = (source_vectors[:, None, :] * self.contexts(negatives)).sum(dim=-1)
        return positive, negative


def weighted_sgns_loss(positive_logits, negative_logits, weights, normalizer):
    per_pair = F.softplus(-positive_logits) + F.softplus(negative_logits).sum(dim=1)
    return (weights * per_pair).mean() / normalizer


def train_sgns(baskets, item_count, config, validate_epoch=None, on_epoch=None, on_best=None):
    config = validate_config(config)
    validate_integer(item_count, 'item_count')
    if not isinstance(baskets, BasketTable) or baskets.items.max() >= item_count:
        raise ValueError('SGNS requires compact baskets within the catalog')
    weights, normalizer = basket_weights(baskets.sizes, config['basket_pair_budget'])
    active, mapped = np.unique(baskets.items, return_inverse=True)
    training = BasketTable(mapped.astype(np.int32), baskets.offsets)
    counts = np.bincount(training.items, minlength=len(active))
    device = config['device']
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(config['seed'])
    model = SGNSModel(len(active), config['vector_size']).to(device)
    optimizer = torch.optim.SparseAdam(model.parameters(), lr=.002)
    sampler = NegativeSampler(counts, .5, config['seed'] + 1, device=device)
    pairs_per_basket = training.sizes * (training.sizes - 1)
    pair_count = int(pairs_per_basket.sum())
    contribution = {str(int(size)): float((weights[training.sizes == size] *
                    pairs_per_basket[training.sizes == size]).sum()) for size in np.unique(training.sizes)}
    history, best_vectors, best_contexts = [], None, None
    best_score, best_epoch, waiting, seen_pairs = -float('inf'), 0, 0, 0
    metadata = {**config, 'trainer': 'sgns', 'device': device, 'ns_exponent': .5,
                'torch_version': str(torch.__version__), 'active_items': len(active),
                'final_train_counts_sha256': hashlib.sha256(counts.astype('<i8').tobytes()).hexdigest(),
                'selection_mode': 'validation' if validate_epoch else 'fixed_epochs'}
    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()

    def export():
        vectors = np.zeros((item_count, config['vector_size']), dtype=np.float32)
        contexts = np.zeros_like(vectors)
        vectors[active] = model.inputs.weight.detach().cpu().numpy()
        contexts[active] = model.contexts.weight.detach().cpu().numpy()
        return vectors, contexts

    for epoch in range(1, config['epochs'] + 1):
        started = time.monotonic()
        loss_sum, trained_pairs = 0., 0
        statistics = dict(negative_count=0, hard_negative_count=0, resampled_count=0,
                          fallback_count=0, hard_random_fallback_count=0)
        for batch in iter_pair_batches(training, config['basket_pair_budget'], config['pair_batch_size'],
                                        config['seed'], epoch):
            negatives, stats = sampler.sample(
                batch['source_ids'], batch['basket_ids'], training, config['negative'],
                exclusion=config['negative_exclusion'], hard_ratio=config['hard_negative_ratio'],
                model=model, epoch=epoch, warmup_epochs=config['hard_negative_warmup'],
                pool_size=config['hard_negative_pool_size'], target_ids=batch['target_ids'])
            sources = torch.as_tensor(batch['source_ids'], dtype=torch.long, device=device)
            targets = torch.as_tensor(batch['target_ids'], dtype=torch.long, device=device)
            pair_weights = torch.as_tensor(batch['weights'], dtype=torch.float32, device=device)
            rate = .002 + (.0005 - .002) * seen_pairs / (pair_count * config['epochs'])
            for group in optimizer.param_groups:
                group['lr'] = rate
            optimizer.zero_grad(set_to_none=True)
            positives, negative_logits = model(sources, targets, negatives)
            loss = weighted_sgns_loss(positives, negative_logits, pair_weights, normalizer)
            if not torch.isfinite(loss).item():
                raise ValueError('nonfinite SGNS loss')
            loss.backward()
            if any(not torch.isfinite(parameter.grad.coalesce().values()).all().item()
                   for parameter in model.parameters() if parameter.grad is not None):
                raise ValueError('nonfinite SGNS gradient')
            optimizer.step()
            amount = len(sources)
            trained_pairs += amount
            seen_pairs += amount
            loss_sum += loss.item() * amount
            for name in statistics:
                statistics[name] += stats[name]
        if trained_pairs != pair_count:
            raise RuntimeError('SGNS pair generation did not cover the complete training epoch')
        training_seconds = time.monotonic() - started
        validation_start = time.monotonic()
        metrics, vectors, contexts = {}, None, None
        if validate_epoch:
            vectors, contexts = export()
            metrics = validate_epoch(vectors, contexts, epoch)
            if any(value is not None and (not isinstance(value, (int, float, np.number))
                   or not np.isfinite(value)) for value in metrics.values()):
                raise ValueError('validation metrics must be finite numbers or None')
            score = metrics.get('warm_macro_recall_at_50')
            if not isinstance(score, (float, int)) or not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError('validation must return a finite warm_macro_recall_at_50 in [0, 1]')
            improved = score > best_score + config['min_delta']
        else:
            score, improved = None, epoch == config['epochs']
        if improved:
            best_score, best_epoch, waiting = score, epoch, 0
            if vectors is None:
                vectors, contexts = export()
            best_vectors, best_contexts = vectors.copy(), contexts.copy()
            if on_best:
                on_best(best_vectors, best_contexts,
                        {**metadata, 'best_epoch': best_epoch, 'trained_epochs': epoch})
        else:
            waiting += 1
        record = {'epoch': epoch, 'loss': loss_sum / trained_pairs, 'pair_count': trained_pairs,
                  'normalizer': normalizer, 'basket_contribution': contribution,
                  'actual_hard_negative_ratio': statistics['hard_negative_count']/statistics['negative_count'],
                  **statistics, 'learning_rate': rate, 'training_seconds': training_seconds,
                  'validation_seconds': time.monotonic() - validation_start,
                  'pairs_per_second': trained_pairs / max(training_seconds, 1e-9),
                  'process_peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss *
                    (1 if sys.platform == 'darwin' else 1024),
                  'peak_gpu_memory_bytes': torch.cuda.max_memory_allocated() if device == 'cuda' else None,
                  'best': improved, 'metrics': metrics}
        history.append(record)
        if on_epoch:
            on_epoch(record)
        if validate_epoch and waiting >= config['patience']:
            break
    metadata.update(best_epoch=best_epoch, trained_epochs=epoch)
    return {'vectors': best_vectors, 'context_vectors': best_contexts,
            'metadata': metadata, 'history': history}
