import math

import numpy as np
import pytest
import torch


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def data():
    from item2vec.sampling import BasketTable
    return BasketTable(np.tile([0, 1, 2, 3], 10), np.arange(0, 41, 2))


def test_weighted_loss_has_correct_scale_and_scales_both_gradients():
    from item2vec.sgns import weighted_sgns_loss
    positives = torch.zeros(2, requires_grad=True)
    negatives = torch.zeros((2, 2), requires_grad=True)
    loss = weighted_sgns_loss(positives, negatives, torch.tensor([1., .1]), .55)
    assert loss.item() == pytest.approx(3*math.log(2))
    loss.backward()
    assert positives.grad[1]/positives.grad[0] == pytest.approx(.1)
    assert negatives.grad[1, 0]/negatives.grad[0, 0] == pytest.approx(.1)


def test_best_epoch_is_returned_after_early_stopping():
    from item2vec.sgns import train_sgns
    snapshots, events, checkpoints = [], [], []
    scores = [.10, .12, .11, .105, .10]
    def validate(vectors, contexts, epoch):
        snapshots.append(vectors.copy())
        return {'warm_macro_recall_at_50': scores[epoch-1]}
    result = train_sgns(data(), 5, {'vector_size': 8, 'epochs': 10, 'negative': 2,
                        'seed': 17, 'pair_batch_size': 64, 'device': 'cpu'},
                        validate_epoch=validate, on_epoch=events.append,
                        on_best=lambda v, c, m: checkpoints.append(m['best_epoch']))
    assert result['metadata']['best_epoch'] == 2
    assert result['metadata']['trained_epochs'] == 5
    assert checkpoints == [1, 2]
    np.testing.assert_array_equal(result['vectors'], snapshots[1])
    assert not np.array_equal(result['vectors'], snapshots[-1])
    assert np.isfinite(result['vectors']).all() and np.isfinite(result['context_vectors']).all()
    assert np.any(result['vectors'][:4]) and not np.any(result['vectors'][4])
    assert len(events) == 5 and all(event['pair_count'] == 40 for event in events)


def test_fixed_epochs_reproducibility_and_learning_real_pairs():
    from item2vec.sgns import train_sgns
    config = {'vector_size': 8, 'epochs': 30, 'negative': 3, 'seed': 17,
              'pair_batch_size': 40, 'device': 'cpu'}
    result = train_sgns(data(), 5, config)
    again = train_sgns(data(), 5, config)
    np.testing.assert_array_equal(result['vectors'], again['vectors'])
    assert result['metadata']['selection_mode'] == 'fixed_epochs'
    assert result['metadata']['best_epoch'] == 30
    scores = result['vectors'] @ result['context_vectors'].T
    assert scores[0, 1] > scores[0, 2]
    assert scores[2, 3] > scores[2, 0]


def test_weighted_training_keeps_identical_active_catalog_rows():
    from item2vec.sgns import train_sgns
    from item2vec.sampling import BasketTable
    baskets = BasketTable(np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]),
                          np.array([0, 10, 12]))
    common = {'vector_size': 4, 'epochs': 1, 'negative': 2, 'device': 'cpu'}
    baseline = train_sgns(baskets, 13, common)
    weighted = train_sgns(baskets, 13, {**common, 'basket_pair_budget': 30})
    np.testing.assert_array_equal(np.any(baseline['vectors'] != 0, axis=1),
                                  np.any(weighted['vectors'] != 0, axis=1))
    assert weighted['history'][0]['basket_contribution']['10'] == pytest.approx(30)


def test_real_training_activates_hard_samples_after_warmup():
    from item2vec.sgns import train_sgns
    result = train_sgns(data(),5,{'vector_size':4,'epochs':4,'negative':2,
                                 'hard_negative_ratio':.5,'device':'cpu'})
    assert [row['hard_negative_count'] for row in result['history']]==[0,0,40,40]
    assert np.isfinite(result['vectors']).all() and np.isfinite(result['context_vectors']).all()


@pytest.mark.parametrize('config', [{'negative': 0}, {'hard_negative_ratio': 1},
    {'device': 'mps'}, {'basket_pair_budget': -1}, {'patience': 0}, {'min_delta': float('nan')},
    {'hard_negative_pool_size': 1, 'hard_negative_ratio': .9}, {'typo': 1}])
def test_invalid_config_fails_before_training(config):
    from item2vec.sgns import train_sgns
    with pytest.raises(ValueError):
        train_sgns(data(), 5, config)


def test_nonfinite_validation_stops_before_overwriting_best():
    from item2vec.sgns import train_sgns
    saved = []
    def validate(v, c, epoch):
        return {'warm_macro_recall_at_50': .1 if epoch == 1 else float('nan')}
    with pytest.raises(ValueError, match='validation'):
        train_sgns(data(), 5, {'vector_size': 4, 'epochs': 3, 'negative': 2, 'device': 'cpu'},
                   validate_epoch=validate, on_best=lambda v, c, m: saved.append(m['best_epoch']))
    assert saved == [1]


def test_nonfinite_secondary_metric_cannot_replace_the_best_model():
    from item2vec.sgns import train_sgns
    saved = []
    def validate(v,c,epoch):
        return {'warm_macro_recall_at_50':.1*epoch,
                'weighted_recall_at_50':.1 if epoch==1 else float('nan')}
    with pytest.raises(ValueError,match='validation'):
        train_sgns(data(),5,{'vector_size':4,'epochs':2,'negative':2,'device':'cpu'},
                   validate_epoch=validate,on_best=lambda v,c,m:saved.append(m['best_epoch']))
    assert saved == [1]
