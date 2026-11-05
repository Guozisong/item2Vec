"""Unordered basket context prediction with a shared cosine retrieval space."""

import math
import os
from pathlib import Path
import shutil
import tempfile
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from item2vec.sampling import BasketTable, NegativeSampler, validate_integer


DEFAULTS = dict(vector_size=128, negative=15, epochs=10, attention_heads=4,
                temperature=.1, pair_loss_weight=1., context_batch_size=256,
                learning_rate=.001, seed=1, device='auto', patience=3, min_delta=.0001)


def validate_config(config):
    unknown = set(config) - DEFAULTS.keys()
    if unknown:
        raise ValueError(f'unknown attention options: {sorted(unknown)}')
    config = {**DEFAULTS, **config}
    for name in ('vector_size', 'negative', 'epochs', 'attention_heads', 'context_batch_size', 'patience'):
        validate_integer(config[name], name)
    validate_integer(config['seed'], 'seed', 0)
    for name in ('temperature', 'pair_loss_weight', 'learning_rate', 'min_delta'):
        value = config[name]
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float))
                or not math.isfinite(value) or value < 0):
            raise ValueError(f'{name} must be finite and nonnegative')
    if not config['temperature'] or not config['learning_rate']:
        raise ValueError('temperature and learning_rate must be positive')
    if config['vector_size'] % config['attention_heads']:
        raise ValueError('vector_size must be divisible by attention_heads')
    if config['device'] not in ('auto', 'cpu', 'cuda'):
        raise ValueError('device must be auto, cpu or cuda')
    if config['device'] == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA was requested but is unavailable')
    return config


def iter_context_batches(baskets, batch_size, seed, epoch):
    """Each distinct item is a target once per epoch; remove it before encoding."""
    rng = np.random.default_rng(np.random.SeedSequence([seed, epoch, 0]))
    order = rng.permutation(len(baskets))
    sizes = baskets.sizes
    for start in range(0, len(order), 256):
        owners = order[start:start + 256]
        positions = np.concatenate([np.arange(baskets.offsets[i], baskets.offsets[i+1]) for i in owners])
        owners = np.repeat(owners, sizes[owners])
        permutation = rng.permutation(len(positions))
        for offset in range(0, len(permutation), batch_size):
            take = permutation[offset:offset + batch_size]
            ids = owners[take]
            targets = baskets.items[positions[take]]
            contexts = baskets.padded(ids)
            contexts[contexts == targets[:, None]] = -1
            compact = np.argsort(contexts < 0, axis=1, kind='stable')
            contexts = np.take_along_axis(contexts, compact, axis=1)[:, :-1]
            yield {'targets': targets, 'contexts': contexts, 'basket_ids': ids}


def sample_context_view(contexts, rng):
    """Uniformly sample a nonempty context size, including singleton contexts."""
    valid = contexts >= 0
    sizes = valid.sum(axis=1)
    keep = rng.integers(1, sizes + 1)
    scores = np.where(valid, rng.random(contexts.shape), np.inf)
    ranks = np.argsort(np.argsort(scores, axis=1), axis=1)
    return np.where(valid & (ranks < keep[:, None]), contexts, -1)


class AttentionItem2Vec(nn.Module):
    def __init__(self, item_count, dimension, heads):
        super().__init__()
        self.padding = item_count
        self.embedding = nn.Embedding(item_count + 1, dimension, padding_idx=item_count)
        nn.init.normal_(self.embedding.weight, std=.05)
        with torch.no_grad():
            self.embedding.weight[item_count].zero_()
        self.encoder = nn.TransformerEncoderLayer(dimension, heads, dim_feedforward=2 * dimension,
                                                  dropout=0., batch_first=True)

    def item_vectors(self, ids):
        return F.normalize(self.embedding(ids), dim=-1)

    def encode_context(self, contexts):
        valid = contexts >= 0
        if not bool(valid.any(dim=1).all().item()):
            raise ValueError('contexts must be nonempty')
        ids = contexts.masked_fill(~valid, self.padding)
        encoded = self.encoder(self.embedding(ids), src_key_padding_mask=~valid)
        pooled = (encoded * valid[:, :, None]).sum(dim=1) / valid.sum(dim=1, keepdim=True)
        return F.normalize(pooled, dim=-1)


def joint_loss(model, contexts, targets, negatives, temperature, pair_weight, context_view=None):
    target = model.item_vectors(targets)
    negative = model.item_vectors(negatives)
    def context_loss(context):
        query = model.encode_context(context)
        positive = (query * target).sum(dim=-1, keepdim=True)
        noise = torch.einsum('bd,bkd->bk', query, negative)
        logits = torch.cat((positive, noise), dim=1) / temperature
        return F.cross_entropy(logits, torch.zeros(len(targets), dtype=torch.long, device=targets.device))
    context = context_loss(contexts)
    if context_view is not None:
        context = (context + context_loss(context_view)) / 2
    valid = contexts >= 0
    positive = (model.item_vectors(contexts.masked_fill(~valid, model.padding)) * target[:, None]).sum(-1)
    positive = positive / temperature
    noise = torch.einsum('bd,bkd->bk', target, negative) / temperature
    losses = torch.logaddexp(positive, noise.logsumexp(dim=1, keepdim=True)) - positive
    pair = ((losses * valid).sum(dim=1) / valid.sum(dim=1)).mean()
    return context + pair_weight * pair, context, pair


def train_attention(baskets, item_count, config, validate_epoch=None, on_epoch=None, on_best=None):
    config = validate_config(config)
    validate_integer(item_count, 'item_count')
    if not isinstance(baskets, BasketTable) or baskets.items.max() >= item_count:
        raise ValueError('attention requires compact baskets within the catalog')
    active, mapped = np.unique(baskets.items, return_inverse=True)
    training = BasketTable(mapped.astype(np.int32), baskets.offsets)
    counts = np.bincount(training.items, minlength=len(active))
    device = config['device']
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(config['seed'])
    model = AttentionItem2Vec(len(active), config['vector_size'], config['attention_heads']).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=0.)
    sampler = NegativeSampler(counts, .5, config['seed'] + 1, device=device)
    metadata = {**config, 'trainer': 'attention', 'behavior_score': 'cosine',
                'device': device, 'torch_version': str(torch.__version__), 'ns_exponent': .5,
                'negative_exclusion': 'basket', 'context_views': 'full_and_random_nonempty',
                'active_items': len(active),
                'selection_mode': 'validation' if validate_epoch else 'fixed_epochs'}
    history, best_state, best_vectors = [], None, None
    best_score, best_epoch, waiting, seen = -float('inf'), 0, 0, 0

    def export():
        vectors = np.zeros((item_count, config['vector_size']), dtype=np.float32)
        with torch.no_grad():
            vectors[active] = model.item_vectors(torch.arange(len(active), device=device)).cpu().numpy()
        if not np.isfinite(vectors).all() or np.any(np.linalg.norm(vectors[active], axis=1) < .99):
            raise ValueError('attention exported nonfinite or zero item vectors')
        return vectors

    for epoch in range(1, config['epochs'] + 1):
        model.train()
        started = time.monotonic()
        rng = np.random.default_rng(np.random.SeedSequence([config['seed'], epoch, 1]))
        loss_sum = context_sum = pair_sum = 0.
        trained = 0
        for batch in iter_context_batches(training, config['context_batch_size'], config['seed'], epoch):
            negatives, _ = sampler.sample(batch['targets'], batch['basket_ids'], training, config['negative'])
            targets = torch.as_tensor(batch['targets'], dtype=torch.long, device=device)
            contexts = torch.as_tensor(batch['contexts'], dtype=torch.long, device=device)
            view = torch.as_tensor(sample_context_view(batch['contexts'], rng), dtype=torch.long, device=device)
            rate = config['learning_rate'] * (1 - .75 * seen / (len(training.items) * config['epochs']))
            for group in optimizer.param_groups:
                group['lr'] = rate
            optimizer.zero_grad(set_to_none=True)
            loss, context, pair = joint_loss(model, contexts, targets, negatives,
                                              config['temperature'], config['pair_loss_weight'], view)
            if not torch.isfinite(loss).item():
                raise ValueError('nonfinite attention loss')
            loss.backward()
            if any(not torch.isfinite(p.grad).all().item() for p in model.parameters() if p.grad is not None):
                raise ValueError('nonfinite attention gradient')
            optimizer.step()
            amount = len(targets)
            trained += amount
            seen += amount
            loss_sum += loss.item() * amount
            context_sum += context.item() * amount
            pair_sum += pair.item() * amount
        if trained != len(training.items):
            raise RuntimeError('attention epoch did not cover all targets')
        training_seconds = time.monotonic() - started
        model.eval()
        vectors = export()
        started = time.monotonic()
        metrics = validate_epoch(vectors, None, epoch) if validate_epoch else {}
        if any(v is not None and (not isinstance(v, (int, float, np.number)) or not np.isfinite(v))
               for v in metrics.values()):
            raise ValueError('validation metrics must be finite numbers or None')
        score = metrics.get('warm_macro_recall_at_50') if validate_epoch else None
        if validate_epoch and (not isinstance(score, (float, int)) or not math.isfinite(score) or not 0 <= score <= 1):
            raise ValueError('validation requires warm_macro_recall_at_50 in [0, 1]')
        improved = score > best_score + config['min_delta'] if validate_epoch else epoch == config['epochs']
        if improved:
            best_score, best_epoch, waiting = score, epoch, 0
            best_vectors = vectors.copy()
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            if on_best:
                on_best(best_vectors, None, {**metadata, 'best_epoch': epoch, 'trained_epochs': epoch})
        else:
            waiting += 1
        record = {'epoch': epoch, 'loss': loss_sum / trained, 'context_loss': context_sum / trained,
                  'pair_loss': pair_sum / trained, 'target_count': trained, 'learning_rate': rate,
                  'training_seconds': training_seconds, 'validation_seconds': time.monotonic() - started,
                  'metrics': metrics, 'best': improved}
        history.append(record)
        if on_epoch:
            on_epoch(record)
        if validate_epoch and waiting >= config['patience']:
            break
    metadata.update(best_epoch=best_epoch, trained_epochs=epoch)
    return {'vectors': best_vectors, 'context_vectors': None, 'history': history, 'metadata': metadata,
            'state_dict': best_state, 'active_indexes': active}


def save_checkpoint(result, path):
    """Save selected encoder and embeddings without seeking on the output mount."""
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    payload = {'state_dict': result['state_dict'], 'metadata': result['metadata'],
               'active_indexes': torch.as_tensor(result['active_indexes'].copy())}
    try:
        with tempfile.TemporaryFile() as stream:
            torch.save(payload, stream)
            stream.seek(0)
            with temporary.open('wb') as output:
                shutil.copyfileobj(stream, output)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
