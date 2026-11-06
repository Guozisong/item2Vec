import numpy as np
import pytest
import torch

from item2vec.sampling import BasketTable


def baskets():
    return BasketTable(np.array([0, 1, 2, 3, 4, 0, 1]), np.array([0, 3, 5, 7]))


def test_context_batches_remove_target_cover_all_items_and_do_not_leak_negatives():
    from item2vec.attention import iter_context_batches
    from item2vec.sampling import NegativeSampler
    data = baskets()
    batches = list(iter_context_batches(data, 2, 17, 1))
    assert sum(len(b['targets']) for b in batches) == len(data.items)
    sampler = NegativeSampler(np.bincount(data.items, minlength=5), seed=17)
    for b in batches:
        assert not np.any(b['contexts'] == b['targets'][:, None])
        for row, owner in enumerate(b['basket_ids']):
            assert set(b['contexts'][row][b['contexts'][row] >= 0]) == set(data[owner]) - {b['targets'][row]}
        negatives, _ = sampler.sample(b['targets'], b['basket_ids'], data, 8)
        for row, owner in enumerate(b['basket_ids']):
            assert not set(negatives[row].tolist()) & set(data[owner])


def test_attention_is_invariant_to_order_and_padding_and_accepts_singletons():
    from item2vec.attention import AttentionItem2Vec
    torch.manual_seed(17)
    model = AttentionItem2Vec(5, 8, 2).eval()
    a = model.encode_context(torch.tensor([[0, 1, -1], [2, -1, -1]]))
    b = model.encode_context(torch.tensor([[1, -1, 0, -1], [-1, 2, -1, -1]]))
    torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-5)
    assert torch.isfinite(a).all()
    with pytest.raises(ValueError, match='nonempty'):
        model.encode_context(torch.tensor([[-1, -1]]))


def test_both_objectives_update_the_exported_shared_vectors():
    from item2vec.attention import AttentionItem2Vec, joint_loss
    for pair_weight in [0., 1.]:
        model = AttentionItem2Vec(5, 8, 2)
        loss, context, pair = joint_loss(model, torch.tensor([[0, 1]]), torch.tensor([2]),
                                         torch.tensor([[3, 4]]), .2, pair_weight)
        loss.backward()
        assert model.embedding.weight.grad[:5].abs().sum() > 0
        assert context.item() > 0 and pair.item() > 0
        assert torch.isfinite(model.embedding.weight.grad).all()
        if pair_weight:
            model.zero_grad()
            joint_loss(model, torch.tensor([[0, 1]]), torch.tensor([2]),
                       torch.tensor([[3, 4]]), .2, pair_weight)[2].backward()
            assert model.embedding.weight.grad[:5].abs().sum() > 0


def test_attention_selects_best_epoch_exports_units_and_preserves_cold_rows():
    from item2vec.attention import train_attention
    snapshots = []
    def validate(v, epoch):
        snapshots.append(v.copy())
        return {'warm_macro_recall_at_50': [.2, .3, .1][epoch-1]}
    result = train_attention(baskets(), 6, {'vector_size': 8, 'attention_heads': 2,
                             'epochs': 3, 'negative': 2, 'context_batch_size': 3,
                             'device': 'cpu', 'patience': 1}, validate_epoch=validate)
    assert result['metadata']['best_epoch'] == 2
    assert result['metadata']['trained_epochs'] == 3
    assert result['metadata']['behavior_score'] == 'cosine'
    np.testing.assert_array_equal(result['vectors'], snapshots[1])
    np.testing.assert_allclose(np.linalg.norm(result['vectors'][:5], axis=1), 1., atol=1e-6)
    assert not result['vectors'][5].any()
    assert all(x['target_count'] == len(baskets().items) for x in result['history'])


@pytest.mark.parametrize('config', [
    {'attention_heads': 0}, {'vector_size': 7, 'attention_heads': 2},
    {'temperature': 0}, {'pair_loss_weight': -1}, {'learning_rate': float('nan')},
    {'negative_exclusion': 'target'}, {'device': 'mps'}, {'context_batch_size': True},
])
def test_attention_rejects_invalid_settings(config):
    from item2vec.attention import validate_config
    with pytest.raises(ValueError):
        validate_config(config)


def test_context_views_include_singletons_without_adding_items():
    from item2vec.attention import sample_context_view
    original = np.tile([0, 1, 2, -1], (100, 1))
    view = sample_context_view(original, np.random.default_rng(17))
    sizes = (view >= 0).sum(axis=1)
    assert sizes.min() == 1 and sizes.max() == 3
    assert np.all((view == original) | (view == -1))


def test_context_iteration_does_not_rescan_all_basket_sizes_per_block():
    from item2vec.attention import iter_context_batches
    class CountedBaskets(BasketTable):
        scans = 0
        @property
        def sizes(self):
            self.scans += 1
            return super().sizes
    data = CountedBaskets(np.tile([0, 1], 1024), np.arange(0, 2049, 2))
    assert sum(len(b['targets']) for b in iter_context_batches(data, 128, 17, 1)) == 2048
    assert data.scans <= 1


def test_attention_learns_group_cosines_and_checkpoint_restores_selected_vectors(tmp_path):
    from item2vec.attention import train_attention, save_checkpoint, AttentionItem2Vec
    data = BasketTable(np.tile([0, 1, 2, 3], 12), np.arange(0, 49, 2))
    result = train_attention(data, 5, {'vector_size': 8, 'attention_heads': 2, 'negative': 3,
                             'context_batch_size': 12, 'epochs': 25, 'learning_rate': .01, 'device': 'cpu'})
    z = result['vectors'][:4]
    assert (z[0] @ z[1] + z[2] @ z[3]) / 2 > np.mean(z[:2] @ z[2:].T) + .3
    path = tmp_path / 'attention.pt'
    save_checkpoint(result, path)
    saved = torch.load(path, weights_only=True)
    model = AttentionItem2Vec(4, 8, 2)
    model.load_state_dict(saved['state_dict'])
    np.testing.assert_allclose(model.item_vectors(torch.arange(4)).detach().numpy(), z, atol=1e-6)
    assert saved['active_indexes'].tolist() == [0, 1, 2, 3]
