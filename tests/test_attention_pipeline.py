import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest


def work_dir(tmp_path):
    from item2vec.preprocessing import preprocess_baskets
    raw, embeddings, processed = [tmp_path / name for name in ('raw', 'embeddings', 'processed')]
    raw.mkdir(); embeddings.mkdir()
    mapping = {str(i): str(i) for i in range(5)}
    (embeddings / 'index2item.json').write_text(json.dumps(mapping))
    (embeddings / 'item2index.json').write_text(json.dumps({str(i): i for i in range(5)}))
    np.eye(5, 768, dtype=np.float32).tofile(embeddings / 'item.feat1CLS')
    pd.DataFrame([(str(o), str(i), '20261001') for o, items in enumerate([[0, 1], [2, 3]] * 3)
                  for i in items], columns=['order_id', 'prod_id', 'dt']).to_csv(raw / 'order_item.csv', index=False)
    preprocess_baskets(raw, embeddings, processed, min_co_basket_orders=1)
    return processed, embeddings


def test_attention_cli_trains_validates_exports_and_queries_cosine(tmp_path):
    from item2vec.training import main
    from item2vec.inference import load_trained_artifacts, query_item, export_all
    processed, embeddings = work_dir(tmp_path)
    validation = tmp_path / 'validation.csv'
    pd.DataFrame([('v1', '0', '20261002'), ('v1', '1', '20261002'),
                  ('v2', '2', '20261002'), ('v2', '3', '20261002')],
                 columns=['order_id', 'prod_id', 'dt']).to_csv(validation, index=False)
    main([str(processed), str(embeddings), '--trainer', 'attention', '--vector-size', '8',
          '--attention-heads', '2', '--epochs', '3', '--negative', '2', '--min-order-count', '1',
          '--context-batch-size', '4', '--device', 'cpu', '--validation-orders', str(validation)])
    artifacts = load_trained_artifacts(embeddings, include_score=True)
    assert len(load_trained_artifacts(embeddings)) == 6
    _, _, vectors, _, contexts, bias, score = artifacts
    assert score == 'cosine' and contexts is None and bias is None
    with np.load(embeddings / 'behavior_item.npz') as archive:
        metadata = json.loads(archive['metadata'].item())
    assert metadata['trainer'] == 'attention' and metadata['behavior_score'] == 'cosine'
    assert (embeddings / 'attention_model.pt').is_file()
    assert (tmp_path / 'validation.json').is_file() and (tmp_path / 'history.json').is_file()
    path = query_item(embeddings, '0', 3, recall_mode='complement', text_weight=0)
    predictions = pd.read_csv(path, dtype={'slave_prod_id': str})
    for row in predictions.itertuples():
        assert row.similarity == pytest.approx(float(vectors[0] @ vectors[int(row.slave_prod_id)]), abs=1e-6)
    exported = pd.read_csv(export_all(embeddings, 3, recall_mode='complement', text_weight=0))
    assert not (exported.master_prod_id == exported.slave_prod_id).any()


@pytest.mark.parametrize('options', [
    ['--trainer', 'sgns', '--attention-heads', '2'],
    ['--trainer', 'gensim', '--temperature', '.1'],
    ['--trainer', 'attention', '--negative-exclusion', 'target'],
    ['--trainer', 'attention', '--basket-pair-budget', '30'],
    ['--trainer', 'attention', '--hard-negative-ratio', '.2'],
    ['--trainer', 'attention', '--vector-size', '7', '--attention-heads', '2'],
])
def test_training_rejects_mixed_or_invalid_attention_options_before_io(options):
    from item2vec.training import main
    with pytest.raises((ValueError, SystemExit)):
        main(['missing-processed', 'missing-embeddings', '--check-options', *options])


def test_cosine_ranker_uses_raw_cosine_and_normalizes_transferred_behavior():
    from item2vec.inference import rank_items
    behavior = np.array([[1., 0.], [.8, .6], [-1., 0.], [0., 0.]])
    text = np.array([[1., 0.], [1., 0.], [-1., 0.], [1., 0.]])
    index = {str(i): str(i) for i in range(4)}
    out = rank_items(text, index, [0, 3], 2, behavior_vectors=behavior,
                     behavior_order_counts=np.ones(4, dtype=np.int64), text_weight=0,
                     recall_mode='complement', behavior_score='cosine',
                     item_categories={str(i): 'category' for i in range(4)})
    direct = out[out.master_prod_id == '0']
    assert direct.similarity.tolist() == pytest.approx([.8, -1.])
    transfer = out[out.master_prod_id == '3']
    assert set(transfer.recall_source) == {'behavior_transfer'}
    proxy = (behavior[0] + behavior[1]) / 2
    proxy /= np.linalg.norm(proxy)
    expected = sorted((float(proxy @ vector) for vector in behavior[:3]), reverse=True)[:2]
    assert transfer.similarity.tolist() == pytest.approx(expected)


def test_cosine_artifact_rejects_legacy_contexts_and_unknown_score(tmp_path):
    from item2vec.inference import load_trained_artifacts
    _, embeddings = work_dir(tmp_path)
    for score in ['cosine', 'unknown']:
        np.savez(embeddings / 'behavior_item.npz', vectors=np.ones((5, 2)),
                 context_vectors=np.ones((5, 2)), item_ids=np.array([str(i) for i in range(5)]),
                 order_counts=np.ones(5, dtype=np.int64),
                 metadata=json.dumps({'trainer': 'attention', 'behavior_score': score}))
        with pytest.raises(ValueError):
            load_trained_artifacts(embeddings)


def test_attention_bash_training_and_query_commands(tmp_path):
    _, embeddings = work_dir(tmp_path)
    root = Path(__file__).resolve().parents[1]
    env = {**os.environ, 'PATH': str(Path(sys.executable).parent) + os.pathsep + os.environ['PATH'],
           'MIN_ORDER_COUNT': '1', 'OMP_NUM_THREADS': '1', 'PYTHONDONTWRITEBYTECODE': '1'}
    commands = [
        ['bash', 'scripts/train.sh', '8', '30', '2', '2', '--trainer', 'attention',
         '--attention-heads', '2', '--context-batch-size', '4', '--temperature', '.2',
         '--pair-loss-weight', '1', '--learning-rate', '.001', '--device', 'cpu', '--work-dir', str(tmp_path)],
        ['bash', 'scripts/query_similar.sh', '0', '3', 'complement', '0', '--work-dir', str(tmp_path)],
        ['bash', 'scripts/export_similarities.sh', '3', '2', 'complement', '0', '--work-dir', str(tmp_path)],
    ]
    for command in commands:
        result = subprocess.run(command, cwd=root, env=env, capture_output=True, text=True)
        assert result.returncode == 0, result.stdout + result.stderr
    assert (embeddings / 'attention_model.pt').is_file()
    assert (tmp_path / 'results/query_0.csv').is_file()
    assert (tmp_path / 'results/item_similarity_complement.csv').is_file()


def test_pipeline_forwards_attention_options_and_rejects_invalid_config_before_writes(tmp_path):
    root = Path(__file__).resolve().parents[1]
    scripts = tmp_path / 'scripts'; scripts.mkdir()
    (scripts / 'run_pipeline.sh').write_text((root / 'scripts/run_pipeline.sh').read_text())
    marker = tmp_path / 'calls'
    for name in ['fetch_data', 'generate_embeddings', 'preprocess_baskets', 'train']:
        (scripts / (name + '.sh')).write_text('printf "%s\\n" "$@" >> "' + str(marker) + '"\n')
    env = {**os.environ, 'PATH': str(Path(sys.executable).parent) + os.pathsep + os.environ['PATH'],
           'PYTHONPATH': str(root / 'src')}
    command = ['bash', str(scripts / 'run_pipeline.sh'), '--trainer', 'attention',
               '--attention-heads', '3', '--context-batch-size', '8']
    result = subprocess.run(command, env=env, capture_output=True, text=True)
    assert result.returncode != 0 and not marker.exists()
    command[command.index('3')] = '4'
    result = subprocess.run(command, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert '--context-batch-size\n8' in marker.read_text()
    assert '--attention-heads\n4' in marker.read_text()
