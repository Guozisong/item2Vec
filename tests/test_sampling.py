import numpy as np
import pytest


def table():
    from item2vec.sampling import BasketTable
    return BasketTable(np.array([0, 1, 1, 2, 3], dtype=np.int32),
                       np.array([0, 2, 5], dtype=np.int64))


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
        np.array([0]), np.array([0]), baskets, k=10000)
    values, counts = np.unique(negatives.numpy(), return_counts=True)
    assert set(values) == {3, 4}
    assert counts[0]/counts.sum() == pytest.approx(np.sqrt(40)/(np.sqrt(40)+np.sqrt(20)), abs=.025)
    assert stats['negative_count'] == 10000


def test_conditional_fallback_and_no_legal_negative():
    from item2vec.sampling import BasketTable, NegativeSampler
    baskets = BasketTable(np.array([0, 1]), np.array([0, 2]))
    sampler = NegativeSampler(np.array([10**12, 10**12, 1]), .5, seed=17)
    negatives, stats = sampler.sample(np.array([0]), np.array([0]), baskets, k=15)
    assert set(negatives.numpy().ravel()) == {2} and stats['fallback_count'] > 0
    with pytest.raises(ValueError, match='legal negative'):
        NegativeSampler(np.array([1, 1]), .5).sample(np.array([0]), np.array([0]), baskets, k=15)


@pytest.mark.parametrize('counts', [[0, 0], [-1, 2], [1.2, 2], [np.nan, 2]])
def test_invalid_negative_frequencies(counts):
    from item2vec.sampling import NegativeSampler
    with pytest.raises(ValueError):
        NegativeSampler(np.array(counts), .5)
