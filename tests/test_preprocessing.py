import csv
import json
import zipfile

import numpy as np
import pytest

from item2vec.preprocessing import preprocess_baskets, load_processed_baskets
from item2vec import preprocessing


def _fixture(tmp_path, rows):
    raw = tmp_path / 'raw'
    embeddings = tmp_path / 'embeddings'
    processed = tmp_path / 'processed'
    raw.mkdir()
    embeddings.mkdir()
    with (raw / 'order_item.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(('order_id', 'prod_id', 'dt'))
        writer.writerows(rows)
    (embeddings / 'index2item.json').write_text(
        json.dumps({'0': 'A', '1': 'B', '2': 'C', '3': 'D'})
    )
    (embeddings / 'item2index.json').write_text(
        json.dumps({'A': 0, 'B': 1, 'C': 2, 'D': 3})
    )
    return raw, embeddings, processed


def test_preprocess_groups_nonadjacent_rows_and_filters_in_two_passes(tmp_path):
    raw, embeddings, processed = _fixture(tmp_path, [
        ('one', 'B', '20261002'), ('two', 'C', '20261001'),
        ('one', 'A', '20261001'), ('two', 'A', '20261001'),
        ('one', 'B', '20261002'), ('one', 'outside', '20261002'),
        ('three', 'A', '20261001'), ('three', 'D', '20261001'),
        ('four', 'B', '20261001'), ('four', 'C', '20261001'),
        ('five', 'A', '20261001'),
    ])

    report = preprocess_baskets(raw, embeddings, processed, max_basket_size=3,
                                min_co_basket_orders=2)
    baskets, all_counts, support, manifest = load_processed_baskets(processed, embeddings)

    assert baskets == [['1', '2'], ['0', '1'], ['0', '2']]
    np.testing.assert_array_equal(all_counts, [4, 2, 2, 1])
    np.testing.assert_array_equal(support, [3, 2, 2, 1])
    assert report == manifest['counts']
    assert report['orders'] == 5
    assert report['duplicate_product_rows'] == 1
    assert report['unknown_product_rows'] == 1
    assert report['cross_date_orders'] == 1
    assert report['initial_valid_baskets'] == 4
    assert report['final_baskets'] == 3
    first = json.loads((processed / 'baskets.jsonl').read_text().splitlines()[0])
    assert first == {'order_id': 'four', 'dt': '20261001', 'items': [1, 2]}


def test_preprocess_drops_large_and_empty_catalog_baskets(tmp_path):
    raw, embeddings, processed = _fixture(tmp_path, [
        ('large', 'A', '20261001'), ('large', 'B', '20261001'),
        ('large', 'C', '20261001'), ('unknown', 'outside', '20261001'),
        ('good', 'A', '20261001'), ('good', 'B', '20261001'),
    ])

    report = preprocess_baskets(raw, embeddings, processed, max_basket_size=2,
                                min_co_basket_orders=1)

    assert report['large_baskets'] == 1
    assert report['zero_known_baskets'] == 1
    assert report['final_baskets'] == 1


def test_minimum_applies_after_dedup_and_after_support_filter(tmp_path):
    rows = [(order, item, '20261001') for order in ('one', 'two') for item in 'ABC']
    rows += [('rare', item, '20261001') for item in 'ABD']
    rows += [('small', item, '20261001') for item in ('A', 'B', 'B', 'outside')]
    raw, embeddings, processed = _fixture(tmp_path, rows)
    report = preprocess_baskets(raw, embeddings, processed, max_basket_size=3,
                                min_co_basket_orders=2, min_basket_size=3)
    baskets, all_counts, support, manifest = load_processed_baskets(processed, embeddings)
    assert baskets == [['0', '1', '2'], ['0', '1', '2']]
    np.testing.assert_array_equal(all_counts, [4, 4, 2, 1])
    np.testing.assert_array_equal(support, [3, 3, 2, 1])
    assert manifest['parameters']['min_basket_size'] == 3
    assert report['small_baskets'] == 1
    assert report['initial_valid_baskets'] == 3
    assert report['dropped_after_support'] == 1
    assert report['final_baskets'] == 2


@pytest.mark.parametrize('minimum', [0, 1, -1, 4, True, 2.5])
def test_invalid_minimum_rejected(tmp_path, minimum):
    raw, embeddings, processed = _fixture(tmp_path, [('one', 'A', '20261001')])
    with pytest.raises(ValueError, match='min_basket_size'):
        preprocess_baskets(raw, embeddings, processed, max_basket_size=3,
                           min_basket_size=minimum)
    assert not processed.exists()


def test_loader_accepts_legacy_minimum_and_rejects_below_manifest_minimum(tmp_path):
    raw, embeddings, processed = _fixture(tmp_path, [
        ('one', 'A', '20261001'), ('one', 'B', '20261001'),
    ])
    preprocess_baskets(raw, embeddings, processed, min_co_basket_orders=1)
    path = processed / 'manifest.json'
    manifest = json.loads(path.read_text())
    del manifest['parameters']['min_basket_size']
    path.write_text(json.dumps(manifest))
    assert load_processed_baskets(processed, embeddings)[0] == [['0', '1']]
    manifest['parameters']['min_basket_size'] = 3
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='invalid items'):
        load_processed_baskets(processed, embeddings)


def test_processed_baskets_reject_changed_orders_or_index(tmp_path):
    raw, embeddings, processed = _fixture(tmp_path, [
        ('one', 'A', '20261001'), ('one', 'B', '20261001'),
    ])
    preprocess_baskets(raw, embeddings, processed, min_co_basket_orders=1)
    with (raw / 'order_item.csv').open('a') as stream:
        stream.write('two,A,20261002\n')
    with pytest.raises(ValueError, match='source orders'):
        load_processed_baskets(processed, embeddings)
    preprocess_baskets(raw, embeddings, processed, min_co_basket_orders=1)
    (embeddings / 'index2item.json').write_text(
        json.dumps({'0': 'B', '1': 'A', '2': 'C', '3': 'D'})
    )
    with pytest.raises(ValueError, match='item index'):
        load_processed_baskets(processed, embeddings)


def test_processed_baskets_reject_parameter_mismatch_and_corruption(tmp_path):
    raw, embeddings, processed = _fixture(tmp_path, [
        ('one', 'A', '20261001'), ('one', 'B', '20261001'),
    ])
    preprocess_baskets(raw, embeddings, processed, min_co_basket_orders=1)
    with pytest.raises(ValueError, match='minimum basket support'):
        load_processed_baskets(processed, embeddings, min_co_basket_orders=5)
    with (processed / 'baskets.jsonl').open('a') as stream:
        stream.write('{}\n')
    with pytest.raises(ValueError, match='manifest'):
        load_processed_baskets(processed, embeddings)


@pytest.mark.parametrize('rows,message', [
    ([('one', 'A', 'bad-date')], 'dt'),
    ([('', 'A', '20261001')], 'order_id'),
    ([('one', '', '20261001')], 'prod_id'),
])
def test_preprocess_rejects_invalid_required_values(tmp_path, rows, message):
    raw, embeddings, processed = _fixture(tmp_path, rows)
    with pytest.raises(ValueError, match=message):
        preprocess_baskets(raw, embeddings, processed)


def test_preprocess_support_works_with_output_that_rejects_seek(monkeypatch, tmp_path):
    raw, embeddings, processed = _fixture(tmp_path, [
        ('one', 'A', '20261001'), ('one', 'B', '20261001'),
    ])
    real_open = preprocessing.Path.open
    writers = []

    class MountedWriter:
        def __init__(self, stream):
            self.stream = stream
            self.seeks = 0

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return self.stream.__exit__(*args)

        def __getattr__(self, name):
            return getattr(self.stream, name)

        def seek(self, *args):
            self.seeks += 1
            if self.seeks > 1:
                raise OSError(22, 'Invalid argument')
            return self.stream.seek(*args)

    def open_output(path, mode='r', *args, **kwargs):
        stream = real_open(path, mode, *args, **kwargs)
        if path == processed / 'support.npz.tmp' and mode == 'wb':
            stream = MountedWriter(stream)
            writers.append(stream)
        return stream

    monkeypatch.setattr(preprocessing.Path, 'open', open_output)
    preprocess_baskets(raw, embeddings, processed, min_co_basket_orders=1)
    baskets, all_counts, support, manifest = load_processed_baskets(processed, embeddings)
    assert baskets == [['0', '1']]
    np.testing.assert_array_equal(all_counts, [1, 1, 0, 0])
    np.testing.assert_array_equal(support, [1, 1, 0, 0])
    assert manifest['counts']['final_baskets'] == 1
    assert len(writers) == 1 and writers[0].seeks == 0
    with zipfile.ZipFile(processed / 'support.npz') as archive:
        assert len(archive.infolist()) == 3
        assert all(entry.flag_bits & 0x08 for entry in archive.infolist())


@pytest.mark.parametrize('failure', ['serialization', 'write'])
def test_support_write_failure_preserves_published_artifacts(monkeypatch, tmp_path, failure):
    raw, embeddings, processed = _fixture(tmp_path, [
        ('one', 'A', '20261001'), ('one', 'B', '20261001'),
    ])
    preprocess_baskets(raw, embeddings, processed, min_co_basket_orders=1)
    published = {name: (processed / name).read_bytes()
                 for name in ('baskets.jsonl', 'support.npz', 'manifest.json')}
    output_streams = []

    def failing_savez(stream, **arrays):
        output_streams.append(stream.stream)
        stream.write(b'partial archive')
        raise OSError('simulated serialization failure')

    original_write = preprocessing._SequentialWriter.write

    def failing_write(writer, data):
        if not output_streams:
            output_streams.append(writer.stream)
            writer.stream.write(data[:16])
            raise OSError('simulated write failure')
        return original_write(writer, data)

    if failure == 'serialization':
        monkeypatch.setattr(preprocessing.np, 'savez', failing_savez)
    else:
        monkeypatch.setattr(preprocessing._SequentialWriter, 'write', failing_write)
    with pytest.raises(OSError, match='simulated'):
        preprocess_baskets(raw, embeddings, processed, min_co_basket_orders=1)
    assert all((processed / name).read_bytes() == content
               for name, content in published.items())
    assert not list(processed.glob('*.tmp'))
    assert len(output_streams) == 1 and output_streams[0].closed
