"""Time-isolated, staged Item2Vec experiments and frozen final evaluation."""

import argparse
from datetime import timedelta
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd

from item2vec.evaluation import BehaviorEvaluator, load_item_categories, split_orders
from item2vec.io import load_index_item, write_json_atomic
from item2vec.preprocessing import _sha256, load_processed_baskets, preprocess_baskets
from item2vec.sampling import validate_integer
from item2vec.scoring import validate_candidate_bias_weight


SEEDS = (17, 42, 2026)
PRIMARY = 'warm_macro_recall_at_50'
THRESHOLDS = dict(macro_relative_gain=.03, max_relative_regression=.01,
                  min_low_frequency_sources=20, min_delta=.0001)


def _code_fingerprint():
    return {path.name: _sha256(path) for path in sorted(Path(__file__).parent.glob('*.py'))}


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _versions():
    versions = {}
    for package in ('torch','gensim','numpy','pandas'):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _freeze(value, path):
    path = Path(path)
    if path.exists() or path.with_suffix('.sha256').exists():
        raise ValueError(f'{path.name} is already frozen')
    write_json_atomic(value, path)
    path.with_suffix('.sha256').write_text(_sha256(path), encoding='utf-8')


def _frozen(path):
    path = Path(path)
    seal = path.with_suffix('.sha256')
    if not path.is_file() or not seal.is_file() or seal.read_text().strip() != _sha256(path):
        raise ValueError(f'{path.name} is missing or its frozen contents changed')
    return _read(path)


def _protocol(root, include_test=False):
    root = Path(root)
    protocol = _frozen(root/'protocol.json')
    if protocol['code'] != _code_fingerprint():
        raise ValueError('experiment implementation changed; prepare a new experiment')
    if protocol['versions'] != _versions():
        raise ValueError('experiment dependency versions changed; prepare a new experiment')
    for relative, fingerprint in protocol['files'].items():
        if relative == 'split/test.csv' and not include_test:
            continue
        if not (root/relative).is_file() or _sha256(root/relative) != fingerprint:
            raise ValueError(f'experiment input changed: {relative}')
    return protocol


def prepare_experiment(work_dir, experiment_dir, validation_days=14, test_days=14,
                       candidate_bias_weight=None, vector_size=512, epochs=10, negative=15,
                       min_co_basket_orders=5, device='auto'):
    work, root = Path(work_dir), Path(experiment_dir)
    for name, value in [('validation_days',validation_days),('test_days',test_days),
                        ('vector_size',vector_size),('epochs',epochs),('negative',negative),
                        ('min_co_basket_orders',min_co_basket_orders)]:
        validate_integer(value,name)
    if device not in ('auto','cpu','cuda'):
        raise ValueError('device must be auto, cpu or cuda')
    if root.exists():
        raise FileExistsError(f'experiment directory already exists: {root}')
    if candidate_bias_weight is None:
        artifact = work/'embeddings'/'behavior_item.npz'
        if not artifact.is_file():
            raise ValueError('specify candidate_bias_weight explicitly when no baseline metadata is available')
        with np.load(artifact,allow_pickle=False) as saved:
            candidate_bias_weight = json.loads(saved['metadata'].item()).get('candidate_bias_weight')
        if candidate_bias_weight is None:
            raise ValueError('baseline metadata has no candidate_bias_weight; specify it explicitly')
    candidate_bias_weight = validate_candidate_bias_weight(candidate_bias_weight)
    required = [work/'raw'/'order_item.csv',work/'raw'/'item.csv',
                work/'embeddings'/'index2item.json',work/'embeddings'/'item2index.json']
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    rows = pd.read_csv(required[0],dtype=str,keep_default_na=False)
    if (not {'order_id','prod_id','dt'}.issubset(rows.columns) or rows.empty
            or any(rows[key].str.strip().eq('').any() for key in ('order_id','prod_id','dt'))
            or not rows.dt.str.fullmatch(r'\d{8}').all()):
        raise ValueError('source orders require nonblank IDs and YYYYMMDD dates')
    pd.to_datetime(rows.dt.unique(),format='%Y%m%d',errors='raise')
    last = pd.to_datetime(rows.dt.max(),format='%Y%m%d')
    test_start = last - timedelta(days=test_days-1)
    validation_start = test_start - timedelta(days=validation_days)
    parts = split_orders(rows,validation_start.strftime('%Y%m%d'),test_start.strftime('%Y%m%d'))
    if any(part.empty for part in parts):
        raise ValueError('time split produced an empty period')
    root.mkdir(parents=True)
    (root/'split').mkdir()
    for name, frame in zip(('train','validation','test'),parts):
        frame.to_csv(root/'split'/f'{name}.csv',index=False)
    shared = root/'train'
    (shared/'raw').mkdir(parents=True)
    (shared/'embeddings').mkdir()
    shutil.copyfile(root/'split'/'train.csv',shared/'raw'/'order_item.csv')
    shutil.copyfile(work/'raw'/'item.csv',shared/'raw'/'item.csv')
    for name in ('index2item.json','item2index.json','item.feat1CLS'):
        source = work/'embeddings'/name
        if source.is_file():
            shutil.copyfile(source,shared/'embeddings'/name)
    preprocess_baskets(shared/'raw',shared/'embeddings',shared/'processed',30,min_co_basket_orders,2)
    files = [path for path in root.rglob('*') if path.is_file()]
    protocol = {'version':1,'code':_code_fingerprint(),'versions':_versions(),'python':sys.version,
                'source_files':{str(path.resolve()):_sha256(path) for path in required},
                'files':{str(path.relative_to(root)):_sha256(path) for path in files},
                'base':dict(vector_size=vector_size,epochs=epochs,negative=negative,
                            max_basket_size=30,min_order_count=min_co_basket_orders,device=device),
                'candidate_bias_weight':candidate_bias_weight,'text_weight':0.,
                'seeds':list(SEEDS),'thresholds':THRESHOLDS.copy(),
                'periods':{'train_max':parts[0].dt.max(),
                           'validation_min':parts[1].groupby('order_id').dt.max().min(),
                           'test_min':parts[2].groupby('order_id').dt.max().min(),
                           'test_max':parts[2].dt.max()}}
    _freeze(protocol,root/'protocol.json')
    return protocol


def _aggregate(runs):
    means, deviations = {}, {}
    for name in runs[0]['metrics']:
        values = [run['metrics'][name] for run in runs]
        if any(value is None for value in values):
            means[name], deviations[name] = None, None
        else:
            if not np.isfinite(values).all():
                raise ValueError(f'nonfinite experiment metric: {name}')
            means[name] = float(np.mean(values))
            deviations[name] = float(np.std(values,ddof=1)) if len(values)>1 else 0.
    return {'mean':means,'std':deviations}


def _guardrails(candidate,baseline,thresholds):
    reasons = []
    for name in ('weighted_recall_at_50','low_frequency_macro_recall_at_50'):
        if (candidate.get(name) is None or baseline.get(name) is None
                or candidate[name] < baseline[name]*(1-thresholds['max_relative_regression'])):
            reasons.append(name)
    if min(candidate.get('low_frequency_sources',0),baseline.get('low_frequency_sources',0)) < thresholds['min_low_frequency_sources']:
        reasons.append('insufficient_low_frequency_sources')
    if candidate['active_items'] != baseline['active_items']:
        reasons.append('active_items')
    if candidate['top100_candidate_share'] > baseline['top100_candidate_share']+1e-12:
        reasons.append('top100_candidate_share')
    return reasons


def assess_candidate(candidate_runs,baseline_runs,g0_metrics,thresholds=None):
    thresholds = thresholds or THRESHOLDS
    for runs in (candidate_runs,baseline_runs):
        if len(runs)!=3 or {run['config']['seed'] for run in runs}!=set(SEEDS):
            raise ValueError('confirmation requires the three distinct seeds 17, 42, 2026')
    candidate,baseline = _aggregate(candidate_runs),_aggregate(baseline_runs)
    mean,reference = candidate['mean'],baseline['mean']
    reasons = _guardrails(mean,reference,thresholds)
    if mean['weighted_recall_at_50'] < g0_metrics['weighted_recall_at_50']*(1-thresholds['max_relative_regression']):
        reasons.append('weighted_recall_vs_g0')
    gain = mean[PRIMARY]-reference[PRIMARY]
    if gain < (reference[PRIMARY]*thresholds['macro_relative_gain'] if reference[PRIMARY] else thresholds['min_delta']):
        reasons.append('macro_gain')
    if mean[PRIMARY] <= g0_metrics[PRIMARY]+candidate['std'][PRIMARY]:
        reasons.append('inconclusive_vs_g0')
    return {'accepted':not reasons,'reasons':reasons,'candidate':candidate,'baseline':baseline,
            'macro_gain_absolute':gain,
            'macro_gain_relative':gain/reference[PRIMARY] if reference[PRIMARY] else None,
            'refit_epochs':int(np.median([run['best_epoch'] for run in candidate_runs]))}


def _run_id(config):
    return f"{config['recipe']}-d{config['vector_size']}-seed{config['seed']}"


def _model(root,run):
    return root/'runs'/run/'embeddings'/'behavior_item.npz'


def _result(root,run):
    result = _read(root/'runs'/run/'result.json')
    if _sha256(_model(root,run)) != result['model_sha256']:
        raise ValueError(f'completed model changed: {run}')
    return result


def _evaluator(root,protocol,split):
    baskets,_,_,_ = load_processed_baskets(root/'train'/'processed',root/'train'/'embeddings',compact=True)
    index = load_index_item(root/'train'/'embeddings')
    counts = np.bincount(baskets.items,minlength=len(index))
    rows = pd.read_csv(root/'split'/f'{split}.csv',dtype=str,keep_default_na=False)
    categories = load_item_categories(root/'train'/'raw'/'item.csv')
    return BehaviorEvaluator(rows,index,counts,protocol['candidate_bias_weight'],categories)


def _evaluate_artifact(path,evaluator):
    with np.load(path,allow_pickle=False) as artifact:
        expected = [str(evaluator.index2item[str(i)]) for i in range(len(evaluator.index2item))]
        if not np.array_equal(artifact['item_ids'],expected):
            raise ValueError('experiment model item IDs do not match the frozen catalog')
        metadata = json.loads(artifact['metadata'].item())
        return evaluator(artifact['vectors'],artifact['context_vectors']),metadata


def _run(root,protocol,config,evaluator):
    run_id = _run_id(config)
    run = root/'runs'/run_id
    if (run/'result.json').exists():
        result = _result(root,run_id)
        if result['config']!=config:
            raise ValueError(f'completed run configuration changed: {run_id}')
        return result
    if run.exists():
        raise ValueError(f'incomplete run exists: {run}; use a new experiment directory')
    (run/'embeddings').mkdir(parents=True)
    (run/'raw').mkdir()
    for source in (root/'train'/'embeddings').iterdir():
        if source.name in ('index2item.json','item2index.json','item.feat1CLS'):
            shutil.copyfile(source,run/'embeddings'/source.name)
    shutil.copyfile(root/'train'/'raw'/'item.csv',run/'raw'/'item.csv')
    write_json_atomic(config,run/'config.json')
    arguments = [sys.executable,'-m','item2vec.training',str(root/'train'/'processed'),str(run/'embeddings'),
                 '--candidate-bias-weight',str(protocol['candidate_bias_weight'])]
    for name in ('vector_size','epochs','negative','max_basket_size','min_order_count','seed','trainer'):
        arguments.extend(['--'+name.replace('_','-'),str(config[name])])
    if config['trainer']=='sgns':
        for name in ('basket_pair_budget','negative_exclusion','hard_negative_ratio','device'):
            arguments.extend(['--'+name.replace('_','-'),str(config[name])])
        arguments.extend(['--validation-orders',str(root/'split'/'validation.csv')])
    print(f"Training {run_id}; log: {run/'training.log'}",flush=True)
    environment = {**os.environ,'PYTHONPATH':str(Path(__file__).parents[1])+os.pathsep+os.environ.get('PYTHONPATH','')}
    with (run/'training.log').open('w',encoding='utf-8') as log:
        subprocess.run(arguments,stdout=log,stderr=subprocess.STDOUT,env=environment,check=True)
    metrics,metadata = _evaluate_artifact(_model(root,run_id),evaluator)
    result = {'run_id':run_id,'config':config,'metrics':metrics,
              'best_epoch':metadata.get('best_epoch',config['epochs']),
              'model_sha256':_sha256(_model(root,run_id))}
    write_json_atomic(result,run/'result.json')
    return result


def _ablation(protocol):
    common = {**protocol['base'],'seed':17}
    return [{**common,'recipe':'G0','trainer':'gensim'},
            *[{**common,'recipe':recipe,'trainer':'sgns','basket_pair_budget':budget,
               'negative_exclusion':exclusion,'hard_negative_ratio':ratio}
              for recipe,budget,exclusion,ratio in [('A0',0,'target',0),('A1',30,'target',0),
                                                    ('A2',30,'basket',0),('A3',30,'basket',.2)]]]


def _screen(root,protocol,results):
    baseline = _result(root,_run_id(_ablation(protocol)[1]))['metrics']
    return [run for run in results if run['config']['recipe'] not in ('G0','A0')
            and not _guardrails(run['metrics'],baseline,protocol['thresholds'])
            and run['metrics'][PRIMARY]>baseline[PRIMARY]+protocol['thresholds']['min_delta']]


def _best(results):
    best_macro = max(run['metrics'][PRIMARY] for run in results)
    tied = [run for run in results if best_macro-run['metrics'][PRIMARY]<.0001]
    return max(tied,key=lambda run:(run['metrics']['warm_ndcg_at_50'],
                                   -run['config']['vector_size'],-run['best_epoch']))


def run_phase(experiment_dir,phase):
    root = Path(experiment_dir).resolve()
    protocol = _protocol(root)
    if (root/'selection.json').exists():
        if phase=='confirmation':
            return _frozen(root/'selection.json')
        raise ValueError('experiment selection is frozen')
    if phase not in ('ablation','dimensions','confirmation'):
        raise ValueError('phase must be ablation, dimensions or confirmation')
    evaluator = _evaluator(root,protocol,'validation')
    configurations = _ablation(protocol)
    if phase=='ablation':
        return {'runs':[_run(root,protocol,config,evaluator) for config in configurations]}
    initial = [_result(root,_run_id(config)) for config in configurations]
    candidates = _screen(root,protocol,initial)
    if phase=='dimensions':
        results = []
        if candidates:
            config = _best(candidates)['config']
            for dimension in sorted({128,256,512,config['vector_size']}):
                results.append(_run(root,protocol,{**config,'vector_size':dimension},evaluator))
        screening = {'runs':results,'status':'screened' if results else 'no_qualifying_recipe'}
        write_json_atomic(screening,root/'dimensions.json')
        return screening
    if not (root/'dimensions.json').is_file():
        raise ValueError('run the dimensions phase before confirmation')
    dimensions = _read(root/'dimensions.json')['runs']
    # Re-read completed results so altered models cannot be selected from cached metrics.
    dimensions = [_result(root,run['run_id']) for run in dimensions]
    finalists = _screen(root,protocol,initial+dimensions)
    selected,baseline = [],[initial[1]]
    assessment = {'accepted':False,'reasons':['no_qualifying_recipe']}
    if finalists:
        config = _best(finalists)['config']
        selected = [_run(root,protocol,{**config,'seed':seed},evaluator) for seed in SEEDS]
        baseline = [_run(root,protocol,{**configurations[1],'seed':seed},evaluator) for seed in SEEDS]
        assessment = assess_candidate(selected,baseline,initial[0]['metrics'],protocol['thresholds'])
    tested = {run['run_id']:run for run in [initial[0],*baseline,
              *(selected if assessment['accepted'] else [])]}
    selection = {'status':'accepted' if assessment['accepted'] else 'keep_baseline',
                 'protocol_sha256':_sha256(root/'protocol.json'),'assessment':assessment,
                 'production_seed':17,
                 'refit_epochs':assessment.get('refit_epochs') if assessment['accepted'] else protocol['base']['epochs'],
                 'production_config':selected[0]['config'] if assessment['accepted'] else initial[0]['config'],
                 'winner_runs':[run['run_id'] for run in selected] if assessment['accepted'] else [initial[0]['run_id']],
                 'runs':{name:{'model_sha256':run['model_sha256'],
                               'result_sha256':_sha256(root/'runs'/name/'result.json')} for name,run in tested.items()}}
    _freeze(selection,root/'selection.json')
    return selection


def test_selection(experiment_dir):
    root = Path(experiment_dir).resolve()
    selection = _frozen(root/'selection.json')
    protocol = _protocol(root,include_test=True)
    if selection['protocol_sha256']!=_sha256(root/'protocol.json'):
        raise ValueError('selection protocol changed')
    for run,expected in selection['runs'].items():
        if (_sha256(_model(root,run))!=expected['model_sha256']
                or _sha256(root/'runs'/run/'result.json')!=expected['result_sha256']):
            raise ValueError(f'frozen model or result changed: {run}')
    fingerprint = _sha256(root/'selection.json')
    path = root/'test_report.json'
    if path.exists():
        report = _read(path)
        if report['selection_sha256']!=fingerprint:
            raise ValueError('frozen selection changed after final testing')
        return report
    evaluator = _evaluator(root,protocol,'test')
    metrics = {run:_evaluate_artifact(_model(root,run),evaluator)[0] for run in selection['runs']}
    confirmation = None
    if selection['status']=='accepted':
        winners = [_result(root,run) for run in selection['winner_runs']]
        baselines = [_result(root,run) for run in selection['runs'] if run.startswith('A0-')]
        g0 = next(run for run in selection['runs'] if run.startswith('G0-'))
        confirmation = assess_candidate([{**run,'metrics':metrics[run['run_id']]} for run in winners],
                                        [{**run,'metrics':metrics[run['run_id']]} for run in baselines],
                                        metrics[g0],protocol['thresholds'])
    report = {'selection_sha256':fingerprint,'test_sha256':protocol['files']['split/test.csv'],
              'metrics':metrics,'confirmation':confirmation,
              'status':'passed' if confirmation and confirmation['accepted'] else
                       'failed' if confirmation else 'baseline_retained'}
    write_json_atomic(report,path)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command',required=True)
    prepare = commands.add_parser('prepare')
    prepare.add_argument('--work-dir',required=True)
    prepare.add_argument('--experiment-dir',required=True)
    for option,default in [('validation-days',14),('test-days',14),('vector-size',512),
                           ('epochs',10),('negative',15),('min-co-basket-orders',5)]:
        prepare.add_argument('--'+option,type=int,default=default)
    prepare.add_argument('--candidate-bias-weight',type=float)
    prepare.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    run = commands.add_parser('run')
    run.add_argument('--experiment-dir',required=True)
    run.add_argument('--phase',required=True,choices=['ablation','dimensions','confirmation'])
    test = commands.add_parser('test')
    test.add_argument('--experiment-dir',required=True)
    args = vars(parser.parse_args(argv))
    command = args.pop('command')
    result = (prepare_experiment(**args) if command=='prepare' else
              run_phase(**args) if command=='run' else test_selection(**args))
    print(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))
