"""Compact baskets and bounded-memory SGNS pair/negative sampling."""

from dataclasses import dataclass

import numpy as np


def validate_integer(value, name, minimum=1):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')
    return int(value)


@dataclass
class BasketTable:
    items: np.ndarray
    offsets: np.ndarray

    def __post_init__(self):
        items, offsets = np.asarray(self.items), np.asarray(self.offsets)
        if (items.ndim != 1 or offsets.ndim != 1 or len(offsets) < 2
                or not np.issubdtype(items.dtype, np.integer)
                or not np.issubdtype(offsets.dtype, np.integer)
                or offsets[0] != 0 or offsets[-1] != len(items)
                or np.any(np.diff(offsets) < 2) or np.any(items < 0)
                or np.any(items > np.iinfo(np.int32).max)):
            raise ValueError('invalid compact basket storage')
        increasing = np.diff(items) > 0
        increasing[offsets[1:-1] - 1] = True
        if not increasing.all():
            raise ValueError('basket items must be sorted and distinct')
        self.items = items.astype(np.int32, copy=False)
        self.offsets = offsets.astype(np.int64, copy=False)

    def __len__(self):
        return len(self.offsets) - 1

    def __getitem__(self, index):
        return self.items[self.offsets[index]:self.offsets[index + 1]]

    @property
    def sizes(self):
        return np.diff(self.offsets)

    def padded(self, basket_ids):
        basket_ids = np.asarray(basket_ids, dtype=np.int64)
        if basket_ids.ndim != 1 or np.any(basket_ids < 0) or np.any(basket_ids >= len(self)):
            raise ValueError('invalid basket_ids')
        sizes = self.offsets[basket_ids + 1] - self.offsets[basket_ids]
        positions = np.arange(int(sizes.max()) if len(sizes) else 0)
        indexes = self.offsets[basket_ids, None] + positions
        return np.where(positions < sizes[:, None], self.items[indexes.clip(max=len(self.items)-1)], -1)


def basket_weights(sizes, budget):
    validate_integer(budget, 'basket_pair_budget', 0)
    sizes = np.asarray(sizes)
    if (sizes.ndim != 1 or not len(sizes) or not np.issubdtype(sizes.dtype, np.integer)
            or np.any(sizes < 2) or np.any(sizes > 30)):
        raise ValueError('basket sizes must be integers between 2 and 30')
    pairs = sizes.astype(np.int64) * (sizes - 1)
    weights = np.minimum(1., budget / pairs) if budget else np.ones(len(sizes))
    return weights, float(np.sum(pairs * weights) / pairs.sum())


def iter_pair_batches(baskets, budget, batch_size=4096, seed=1, epoch=1):
    validate_integer(batch_size, 'pair_batch_size')
    validate_integer(seed, 'seed', 0)
    validate_integer(epoch, 'epoch')
    weights, _ = basket_weights(baskets.sizes, budget)
    rng = np.random.default_rng(np.random.SeedSequence([seed, epoch, 0]))
    order = rng.permutation(len(baskets))
    sizes = baskets.sizes
    templates = {int(size): np.nonzero(~np.eye(size, dtype=bool)) for size in np.unique(sizes)}
    for start in range(0, len(order), 256):
        block = order[start:start + 256]
        sources, targets, owners = [], [], []
        for size in np.unique(sizes[block]):
            selected = block[sizes[block] == size]
            left, right = templates[int(size)]
            beginnings = baskets.offsets[selected, None]
            sources.append(baskets.items[beginnings + left].ravel())
            targets.append(baskets.items[beginnings + right].ravel())
            owners.append(np.repeat(selected, len(left)))
        sources, targets, owners = map(np.concatenate, (sources, targets, owners))
        permutation = rng.permutation(len(sources))
        for offset in range(0, len(permutation), batch_size):
            take = permutation[offset:offset + batch_size]
            yield {'source_ids': sources[take], 'target_ids': targets[take],
                   'basket_ids': owners[take], 'weights': weights[owners[take]]}


class NegativeSampler:
    def __init__(self, counts, exponent=.5, seed=1, device='cpu'):
        import torch
        counts = np.asarray(counts)
        if (counts.ndim != 1 or not np.issubdtype(counts.dtype, np.integer)
                or np.any(counts < 0) or not np.any(counts > 0)):
            raise ValueError('negative frequencies must be nonnegative integer counts with positive mass')
        if (isinstance(exponent, (bool, np.bool_)) or not isinstance(exponent, (int, float))
                or not np.isfinite(exponent) or exponent < 0):
            raise ValueError('negative exponent must be finite and nonnegative')
        validate_integer(seed, 'seed', 0)
        probabilities = np.where(counts > 0, counts.astype(np.float64) ** exponent, 0.)
        if not np.isfinite(probabilities).all():
            raise ValueError('negative frequencies overflowed')
        self.torch = torch
        self.device = torch.device(device)
        self.probabilities = torch.tensor(probabilities / probabilities.sum(), dtype=torch.float32,
                                          device=self.device)
        self.generator = torch.Generator(device=self.device).manual_seed(seed)

    def _invalid(self, values, blocked):
        return (values[:, :, None] == blocked[:, None, :]).any(dim=2)

    def _raw(self, rows, columns):
        return self.torch.multinomial(self.probabilities, rows * columns, replacement=True,
                                      generator=self.generator).reshape(rows, columns)

    def _draw(self, blocked, count, stats):
        values = self._raw(len(blocked), count)
        invalid = self._invalid(values, blocked)
        for _ in range(8):
            amount = int(invalid.sum().item())
            if not amount:
                return values
            stats['resampled_count'] += amount
            values[invalid] = self._raw(1, amount).ravel()
            invalid = self._invalid(values, blocked)
        for row in self.torch.nonzero(invalid.any(dim=1)).flatten().tolist():
            probability = self.probabilities.clone()
            excluded = blocked[row][blocked[row] >= 0]
            probability[excluded] = 0
            if not bool((probability.sum() > 0).item()):
                raise ValueError('basket has no legal negative candidate')
            amount = int(invalid[row].sum().item())
            values[row, invalid[row]] = self.torch.multinomial(
                probability, amount, replacement=True, generator=self.generator)
            stats['fallback_count'] += amount
        return values

    def sample(self, source_ids, basket_ids, baskets, k, exclusion='basket', hard_ratio=0.,
               model=None, epoch=1, warmup_epochs=2, pool_size=128, target_ids=None):
        validate_integer(k, 'negative')
        validate_integer(epoch, 'epoch')
        validate_integer(warmup_epochs, 'hard_negative_warmup', 0)
        validate_integer(pool_size, 'hard_negative_pool_size')
        if (isinstance(hard_ratio, (bool, np.bool_)) or not isinstance(hard_ratio, (int, float))
                or not np.isfinite(hard_ratio) or not 0 <= hard_ratio < 1):
            raise ValueError('hard_negative_ratio must be finite and in [0, 1)')
        source_ids = np.asarray(source_ids)
        basket_ids = np.asarray(basket_ids)
        if (source_ids.ndim != 1 or not len(source_ids) or source_ids.shape != basket_ids.shape
                or not np.issubdtype(source_ids.dtype, np.integer)
                or np.any(source_ids < 0) or np.any(source_ids >= len(self.probabilities))):
            raise ValueError('invalid source_ids or basket_ids')
        if exclusion == 'basket':
            excluded = baskets.padded(basket_ids)
        elif exclusion == 'target':
            if target_ids is None:
                raise ValueError('target_ids is required for target exclusion')
            excluded = np.asarray(target_ids)
            if (excluded.shape != source_ids.shape or not np.issubdtype(excluded.dtype, np.integer)
                    or np.any(excluded < 0) or np.any(excluded >= len(self.probabilities))):
                raise ValueError('invalid target_ids')
            excluded = excluded[:, None]
        else:
            raise ValueError('negative_exclusion must be target or basket')
        if np.any(excluded >= len(self.probabilities)):
            raise ValueError('basket item outside negative vocabulary')
        blocked = self.torch.as_tensor(excluded, dtype=self.torch.long, device=self.device)
        hard_count = int(k * hard_ratio) if epoch > warmup_epochs else 0
        if hard_count > pool_size:
            raise ValueError('hard_negative_pool_size is smaller than the hard sample count')
        stats = {'negative_count': len(source_ids) * k, 'hard_negative_count': len(source_ids) * hard_count,
                 'resampled_count': 0, 'fallback_count': 0, 'hard_random_fallback_count': 0}
        negatives = self._draw(blocked, k - hard_count, stats)
        if hard_count:
            if model is None:
                raise ValueError('hard negatives require a model')
            pool = self._raw(len(blocked), pool_size)
            with self.torch.no_grad():
                sources = self.torch.as_tensor(source_ids, dtype=self.torch.long, device=self.device)
                scores = model.score_candidates(sources, pool)
                if scores.shape != pool.shape or not self.torch.isfinite(scores).all():
                    raise ValueError('hard candidate scores must be finite and match the pool')
                scores = scores.masked_fill(self._invalid(pool, blocked), float('-inf'))
                chosen_scores, chosen = scores.topk(hard_count, dim=1)
                hard = pool.gather(1, chosen)
                missing = ~self.torch.isfinite(chosen_scores)
                if missing.any():
                    replacements = self._draw(blocked, hard_count, stats)
                    hard[missing] = replacements[missing]
                    replaced = int(missing.sum().item())
                    stats['fallback_count'] += replaced
                    stats['hard_random_fallback_count'] += replaced
                    stats['hard_negative_count'] -= replaced
            negatives = self.torch.cat((negatives, hard), dim=1)
        return negatives, stats
