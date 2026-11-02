from collections import Counter

import numpy as np
import pytest


def table():
    from item2vec.sampling import BasketTable
    return BasketTable(np.array([0, 1, 1, 2, 3], dtype=np.int32),
                       np.array([0, 2, 5], dtype=np.int64))


def test_contribution_budget_and_disabled_weights():
    from item2vec.sampling import basket_weights
    sizes = np.array([2, 5, 10, 20, 30])
    weights, normalizer = basket_weights(sizes, 30)
    pairs = sizes * (sizes - 1)
    np.testing.assert_allclose(pairs * weights, [2, 20, 30, 30, 30])
    assert normalizer == pytest.approx(112 / pairs.sum())
    weights, normalizer = basket_weights(sizes, 0)
    np.testing.assert_array_equal(weights, np.ones(5))
    assert normalizer == 1


@pytest.mark.parametrize('sizes,budget', [([], 30), ([1], 30), ([2.5], 30),
    ([2], -1), ([2], float('nan')), ([2], True), ([2], 1.5)])
def test_invalid_weight_inputs_fail(sizes, budget):
    from item2vec.sampling import basket_weights
    with pytest.raises(ValueError):
        basket_weights(sizes, budget)


def test_pair_batches_cover_each_directed_pair_once_and_are_reproducible():
    from item2vec.sampling import iter_pair_batches
    def collect(seed):
        batches = list(iter_pair_batches(table(), 30, batch_size=3, seed=seed, epoch=1))
        assert all(0 < len(batch['source_ids']) <= 3 for batch in batches)
        return [(int(s), int(t), int(b), float(w)) for batch in batches for s, t, b, w in zip(
            batch['source_ids'], batch['target_ids'], batch['basket_ids'], batch['weights'])]
    pairs = collect(17)
    assert pairs == collect(17)
    assert Counter(pairs) == Counter(collect(42))
    assert len(pairs) == 8 and all(source != target for source, target, _, _ in pairs)
    assert {(s, t) for s, t, b, _ in pairs if b == 0} == {(0, 1), (1, 0)}
    assert {(s, t) for s, t, b, _ in pairs if b == 1} == {
        (1, 2), (1, 3), (2, 1), (2, 3), (3, 1), (3, 2)}
    np.testing.assert_array_equal(table().padded(np.array([1, 0])), [[1, 2, 3], [0, 1, -1]])


@pytest.mark.parametrize('items,offsets', [([0], [0, 1]), ([0, 1], [1, 2]),
    ([0, 1], [0, 3]), ([-1, 1], [0, 2]), ([0, 1], [0, 2, 1])])
def test_basket_table_rejects_invalid_storage(items, offsets):
    from item2vec.sampling import BasketTable
    with pytest.raises(ValueError):
        BasketTable(np.array(items), np.array(offsets))


def test_negatives_exclude_complete_basket_and_keep_count():
    from item2vec.sampling import BasketTable, NegativeSampler
    baskets = BasketTable(np.array([0, 1, 2]), np.array([0, 3]))
    sampler = NegativeSampler(np.array([100, 80, 60, 40, 20]), .5, seed=17)
    negatives, stats = sampler.sample(
        np.array([0]), np.array([0]), baskets, k=10000, exclusion='basket')
    values, counts = np.unique(negatives.numpy(), return_counts=True)
    assert set(values) == {3, 4}
    assert counts[0]/counts.sum() == pytest.approx(np.sqrt(40)/(np.sqrt(40)+np.sqrt(20)), abs=.025)
    assert stats['negative_count'] == 10000


def test_target_exclusion_requires_and_uses_positive_target():
    from item2vec.sampling import NegativeSampler
    sampler = NegativeSampler(np.array([5, 5, 5, 5]), .5, seed=17)
    with pytest.raises(ValueError, match='target_ids'):
        sampler.sample(np.array([0]), np.array([0]), table(), k=15, exclusion='target')
    negatives, _ = sampler.sample(np.array([0]), np.array([0]), table(), k=500,
                                  exclusion='target', target_ids=np.array([1]))
    assert 1 not in negatives.numpy() and 0 in negatives.numpy()


def test_conditional_fallback_and_no_legal_negative():
    from item2vec.sampling import BasketTable, NegativeSampler
    baskets = BasketTable(np.array([0, 1]), np.array([0, 2]))
    sampler = NegativeSampler(np.array([10**12, 10**12, 1]), .5, seed=17)
    negatives, stats = sampler.sample(np.array([0]), np.array([0]), baskets, k=15)
    assert set(negatives.numpy().ravel()) == {2} and stats['fallback_count'] > 0
    with pytest.raises(ValueError, match='legal negative'):
        NegativeSampler(np.array([1, 1]), .5).sample(np.array([0]), np.array([0]), baskets, k=15)


def test_hard_negatives_start_after_warmup_and_exclude_basket():
    import torch
    from item2vec.sampling import NegativeSampler
    class Scores:
        def score_candidates(self, sources, candidates):
            assert not torch.is_grad_enabled()
            return candidates.float()
    sampler = NegativeSampler(np.array([100, 100, 100, 100, 100]), .5, seed=17)
    _, stats = sampler.sample(np.array([0]), np.array([0]), table(), k=15,
                              hard_ratio=.2, model=Scores(), epoch=2, warmup_epochs=2)
    assert stats['hard_negative_count'] == 0
    negatives, stats = sampler.sample(np.array([0]), np.array([0]), table(), k=15,
                                      hard_ratio=.2, model=Scores(), epoch=3, warmup_epochs=2)
    assert stats['hard_negative_count'] == 3
    assert np.all(negatives.numpy()[0, -3:] == 4)
    assert not np.isin(negatives.numpy(), [0, 1]).any()


def test_random_replacements_are_not_counted_as_hard_negatives(monkeypatch):
    import torch
    from item2vec.sampling import NegativeSampler
    sampler = NegativeSampler(np.ones(4,dtype=int),seed=17)
    draw = sampler._raw
    monkeypatch.setattr(sampler,'_raw',lambda rows,columns:
        torch.zeros((rows,columns),dtype=torch.long) if columns==128 else draw(rows,columns))
    class Scores:
        def score_candidates(self,sources,candidates):
            return torch.zeros_like(candidates,dtype=torch.float32)
    negatives,stats = sampler.sample(np.array([0]),np.array([0]),table(),k=15,
                                     hard_ratio=.2,model=Scores(),epoch=3)
    assert stats['hard_negative_count']==0
    assert stats['hard_random_fallback_count']==3
    assert not np.isin(negatives.numpy(),[0,1]).any()


@pytest.mark.parametrize('counts', [[0, 0], [-1, 2], [1.2, 2], [np.nan, 2]])
def test_invalid_negative_frequencies(counts):
    from item2vec.sampling import NegativeSampler
    with pytest.raises(ValueError):
        NegativeSampler(np.array(counts), .5)
