"""Compact baskets and whole-basket negative sampling for attention."""

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

    def sample(self, target_ids, basket_ids, baskets, k):
        validate_integer(k, 'negative')
        target_ids = np.asarray(target_ids)
        basket_ids = np.asarray(basket_ids)
        if (target_ids.ndim != 1 or not len(target_ids) or target_ids.shape != basket_ids.shape
                or not np.issubdtype(target_ids.dtype, np.integer)
                or np.any(target_ids < 0) or np.any(target_ids >= len(self.probabilities))):
            raise ValueError('invalid target_ids or basket_ids')
        excluded = baskets.padded(basket_ids)
        if np.any(excluded >= len(self.probabilities)):
            raise ValueError('basket item outside negative vocabulary')
        blocked = self.torch.as_tensor(excluded, dtype=self.torch.long, device=self.device)
        stats = {'negative_count': len(target_ids) * k, 'resampled_count': 0, 'fallback_count': 0}
        return self._draw(blocked, k, stats), stats
