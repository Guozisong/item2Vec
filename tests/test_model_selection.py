import json

import numpy as np
import pandas as pd
import pytest


def metrics(macro=.2, weighted=.3, low=.15, share=.6):
    return {'warm_macro_recall_at_50': macro, 'weighted_recall_at_50': weighted,
            'low_frequency_macro_recall_at_50': low, 'low_frequency_sources': 100,
            'active_items': 10, 'top100_candidate_share': share, 'warm_ndcg_at_50': .2}


def runs(value):
    return [{'config':{'seed':seed}, 'metrics':value, 'best_epoch':epoch}
            for seed,epoch in zip((17,42,2026),(2,4,3))]


def test_assessment_uses_all_seeds_and_rejects_guardrail_regressions():
    from item2vec.model_selection import assess_candidate
    baseline = runs(metrics())
    result = assess_candidate(runs(metrics(.22)), baseline, metrics())
    assert result['accepted'] and result['refit_epochs']==3
    assert not assess_candidate(runs(metrics(.25, weighted=.28)),baseline,metrics())['accepted']
    assert not assess_candidate(runs(metrics(.25, low=.10)),baseline,metrics())['accepted']
    assert not assess_candidate(runs(metrics(.25, share=.7)),baseline,metrics())['accepted']
    missing = metrics(.25); missing['low_frequency_macro_recall_at_50']=None
    assert not assess_candidate(runs(missing),baseline,metrics())['accepted']
    with pytest.raises(ValueError,match='seeds'):
        assess_candidate(runs(metrics(.25))[:1],baseline,metrics())


def test_one_lucky_seed_does_not_replace_mean_performance():
    from item2vec.model_selection import assess_candidate
    candidate = runs(metrics(.3))
    candidate[1]['metrics']=metrics(.15)
    candidate[2]['metrics']=metrics(.15)
    assert not assess_candidate(candidate,runs(metrics()),metrics())['accepted']


def source_fixture(tmp_path):
    from item2vec.io import write_item_indexes
    work = tmp_path/'source'
    (work/'raw').mkdir(parents=True)
    (work/'embeddings').mkdir()
    ids = ['001','B','C','D','cold']
    write_item_indexes(ids,work/'embeddings')
    np.eye(5,768,dtype=np.float32).tofile(work/'embeddings'/'item.feat1CLS')
    pd.DataFrame([(item,f'{{分类中的商品类，商品品牌为甲，商品名称为{item}}}') for item in ids],
                 columns=['prod_id','prod_description']).to_csv(work/'raw'/'item.csv',index=False)
    rows = [(f't{i}',item,'20261001') for i in range(6) for item in ('001','B','C','D')]
    rows += [(f'v{i}',item,'20261002') for i in range(2) for item in ('001','B','C','D')]
    rows += [(f'x{i}',item,'20261003') for i in range(2) for item in ('001','B','C','D')]
    # Separate the train baskets so that legal negatives exist.
    rows = [(order+('a' if item in ('001','B') else 'b'),item,date) for order,item,date in rows]
    pd.DataFrame(rows,columns=['order_id','prod_id','dt']).to_csv(work/'raw'/'order_item.csv',index=False)
    return work


def test_prepare_freezes_train_only_support_and_protects_existing_directory(tmp_path):
    from item2vec.model_selection import prepare_experiment
    work = source_fixture(tmp_path)
    exp = tmp_path/'experiment'
    protocol = prepare_experiment(work,exp,validation_days=1,test_days=1,
                                  candidate_bias_weight=.4,vector_size=4,epochs=1,negative=2,device='cpu')
    assert protocol['periods']['train_max']=='20261001'
    assert protocol['periods']['validation_min']=='20261002'
    from item2vec.preprocessing import _sha256
    assert protocol['source_files'][str((work/'raw'/'order_item.csv').resolve())]==_sha256(work/'raw'/'order_item.csv')
    manifest = json.loads((exp/'train'/'processed'/'manifest.json').read_text())
    assert manifest['counts']['orders']==12
    assert all(not path.is_symlink() for path in exp.rglob('*'))
    with pytest.raises(FileExistsError):
        prepare_experiment(work,exp,candidate_bias_weight=.4)


def test_changed_input_and_unfrozen_test_are_rejected(tmp_path):
    from item2vec.model_selection import prepare_experiment, run_phase, test_selection
    work = source_fixture(tmp_path)
    exp = tmp_path/'experiment'
    prepare_experiment(work,exp,validation_days=1,test_days=1,candidate_bias_weight=.4)
    with pytest.raises(ValueError,match='selection'):
        test_selection(exp)
    with (exp/'split'/'validation.csv').open('a') as stream:
        stream.write('bad,001,20261002\n')
    with pytest.raises(ValueError,match='changed'):
        run_phase(exp,'ablation')


def test_experiment_cannot_mix_dependency_versions(tmp_path,monkeypatch):
    from item2vec import model_selection as selection
    work = source_fixture(tmp_path)
    exp = tmp_path/'experiment'
    selection.prepare_experiment(work,exp,validation_days=1,test_days=1,candidate_bias_weight=.4)
    original = selection.importlib.metadata.version
    monkeypatch.setattr(selection.importlib.metadata,'version',
                        lambda name:'changed-version' if name=='torch' else original(name))
    with pytest.raises(ValueError,match='versions'):
        selection.run_phase(exp,'ablation')


def test_full_tiny_experiment_and_idempotent_frozen_test(tmp_path,monkeypatch):
    from item2vec.model_selection import prepare_experiment, run_phase, test_selection
    monkeypatch.setenv('OMP_NUM_THREADS','1')
    work = source_fixture(tmp_path)
    exp = tmp_path/'experiment'
    prepare_experiment(work,exp,validation_days=1,test_days=1,candidate_bias_weight=.4,
                       vector_size=4,epochs=1,negative=2,device='cpu')
    test_path = exp/'split'/'test.csv'
    original = pd.read_csv
    def no_test_read(path,*args,**kwargs):
        if str(path)==str(test_path):
            raise AssertionError('training read the test labels')
        return original(path,*args,**kwargs)
    with monkeypatch.context() as scoped:
        scoped.setattr(pd,'read_csv',no_test_read)
        assert len(run_phase(exp,'ablation')['runs'])==5
        run_phase(exp,'dimensions')
        selection = run_phase(exp,'confirmation')
    assert selection['status']=='keep_baseline'
    first = test_selection(exp)
    assert test_selection(exp)==first
    assert first['selection_sha256']
    selection_file = exp/'selection.json'
    selection = json.loads(selection_file.read_text())
    selection['production_seed']=42
    selection_file.write_text(json.dumps(selection))
    with pytest.raises(ValueError,match='changed|frozen'):
        test_selection(exp)


@pytest.mark.parametrize('confirmation_bad',[False,True])
def test_confirmation_freezes_all_winner_seeds_and_excludes_rejected_models(tmp_path,monkeypatch,confirmation_bad):
    from item2vec import model_selection as selection
    from item2vec.io import write_json_atomic
    work = source_fixture(tmp_path)
    exp = tmp_path/'experiment'
    selection.prepare_experiment(work,exp,validation_days=1,test_days=1,candidate_bias_weight=.4)
    def fake_training(root,protocol,config,evaluator):
        run_id = selection._run_id(config)
        directory = root/'runs'/run_id
        (directory/'embeddings').mkdir(parents=True,exist_ok=True)
        selection._model(root,run_id).write_bytes(run_id.encode())
        macro = {'G0':.2,'A0':.2,'A1':.21,'A2':.24,'A3':.23}[config['recipe']]
        if config['recipe']=='A2':
            macro = {128:.25,256:.26,512:.24}[config['vector_size']]
            if confirmation_bad and config['seed']!=17:
                macro=.14
        result = {'run_id':run_id,'config':config,'metrics':metrics(macro),
                  'best_epoch':3,'model_sha256':selection._sha256(selection._model(root,run_id))}
        write_json_atomic(result,directory/'result.json')
        return result
    monkeypatch.setattr(selection,'_run',fake_training)
    selection.run_phase(exp,'ablation')
    selection.run_phase(exp,'dimensions')
    frozen = selection.run_phase(exp,'confirmation')
    if confirmation_bad:
        assert frozen['status']=='keep_baseline'
        assert all(name.startswith(('G0-','A0-')) for name in frozen['runs'])
    else:
        assert frozen['status']=='accepted' and frozen['production_config']['vector_size']==256
        assert len(frozen['winner_runs'])==3 and len(frozen['runs'])==7
        assert frozen['refit_epochs']==3 and frozen['production_seed']==17
