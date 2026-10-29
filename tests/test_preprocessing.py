import csv
import json

import numpy as np
import pytest

from item2vec.preprocessing import preprocess_baskets, load_processed_baskets


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
