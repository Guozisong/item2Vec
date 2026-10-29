import sys
import types

import pandas as pd
import pytest

from item2vec import data_fetch


def csv_inputs(tmp_path):
    items = tmp_path / 'items.csv'
    items.write_text('extra,prod_description,prod_id\nx,商品A,001\n', encoding='utf-8')
    orders = tmp_path / 'orders.csv'
    orders.write_text('dt,prod_id,order_id,extra\n20261001,001,0001,x\n', encoding='utf-8')
    return items, orders


def test_csv_sources_normalize_columns_and_preserve_string_ids_without_credentials(tmp_path, monkeypatch):
    items, orders = csv_inputs(tmp_path)
    monkeypatch.setattr(data_fetch, 'connect_odps', lambda: pytest.fail('CSV must not connect'))
    output = tmp_path / 'raw'
    data_fetch.main([str(output), '--item-input-csv', str(items), '--order-input-csv', str(orders)])
    assert (output / 'item.csv').read_text() == 'prod_id,prod_description\n001,商品A\n'
    assert (output / 'order_item.csv').read_text() == 'order_id,prod_id,dt\n0001,001,20261001\n'


def test_invalid_second_source_preserves_both_existing_outputs(tmp_path):
    items, orders = csv_inputs(tmp_path)
    orders.write_text('user_id,prod_id,dt\nu,001,20261001\n')
    output = tmp_path / 'raw'
    output.mkdir()
    for name in ('item.csv', 'order_item.csv'):
        (output / name).write_text('old')
    with pytest.raises(ValueError, match='order_id'):
        data_fetch.main([str(output), '--item-input-csv', str(items), '--order-input-csv', str(orders)])
    assert all((output / name).read_text() == 'old' for name in ('item.csv', 'order_item.csv'))
    assert not list(output.glob('.fetch-*'))


def test_table_and_query_use_explicit_sql_without_implicit_date_filter(tmp_path, monkeypatch):
    query = tmp_path / 'orders.sql'
    sql = "select order_id, prod_id, dt from history where dt >= '20260101';"
    query.write_text(sql)
    queries = []

    class Reader:
        def open_reader(self, tunnel):
            assert tunnel is True
            return self

        def to_pandas(self, n_process):
            return (pd.DataFrame({'prod_id': ['001'], 'prod_description': ['name']})
                    if len(queries) == 1 else
                    pd.DataFrame({'order_id': ['0001'], 'prod_id': ['001'], 'dt': ['20260102']}))

    class Client:
        def execute_sql(self, statement):
            queries.append(statement)
            return Reader()

    monkeypatch.setattr(data_fetch, 'connect_odps', lambda: Client())
    data_fetch.main([str(tmp_path / 'raw'), '--item-input-table', 'project.catalog',
                     '--order-input-query-file', str(query)])
    assert queries == ['SELECT * FROM project.catalog', sql]


def test_csv_and_odps_sources_can_be_mixed_and_keep_optional_user_id(tmp_path, monkeypatch):
    items, _ = csv_inputs(tmp_path)

    class Reader:
        def execute_sql(self, sql):
            assert sql == 'SELECT * FROM history'
            return self

        def open_reader(self, **kwargs):
            return self

        def to_pandas(self, **kwargs):
            return pd.DataFrame({'dt': ['20261001'], 'user_id': ['u'],
                                 'order_id': ['0001'], 'prod_id': ['001']})

    monkeypatch.setattr(data_fetch, 'connect_odps', lambda: Reader())
    output = tmp_path / 'raw'
    data_fetch.main([str(output), '--item-input-csv', str(items), '--order-input-table', 'history'])
    assert (output / 'order_item.csv').read_text() == 'order_id,user_id,prod_id,dt\n0001,u,001,20261001\n'


@pytest.mark.parametrize('table', ['table;drop table x', 'x where 1=1', 'a..b', '', 'a-b'])
def test_invalid_table_names_rejected(table):
    with pytest.raises(ValueError, match='table name'):
        data_fetch.source_sql(table=table)


def test_empty_query_rejected(tmp_path):
    query = tmp_path / 'query.sql'
    query.write_text('  ')
    with pytest.raises(ValueError, match='empty'):
        data_fetch.source_sql(query_file=query)


@pytest.mark.parametrize('args', [[], ['--item-input-table', 'a'],
    ['--item-input-table', 'a', '--item-input-csv', 'b', '--order-input-table', 'c']])
def test_cli_requires_one_source_per_dataset(tmp_path, args):
    with pytest.raises(SystemExit):
        data_fetch.main([str(tmp_path), *args])


def test_lowercase_odps_settings_supported(monkeypatch):
    for name in ('ALI_ACCESS_ID', 'ALI_SECRET_ACCESS_KEY', 'ALI_PROJECT', 'ALI_ENDPOINT'):
        monkeypatch.delenv(name, raising=False)
    for name, value in dict(access_id='id', access_key='key', project='p', endpoint='https://example.test').items():
        monkeypatch.setenv(name, value)
    captured = []
    monkeypatch.setitem(sys.modules, 'odps', types.SimpleNamespace(ODPS=lambda *args: captured.append(args)))
    data_fetch.connect_odps()
    assert captured == [('id', 'key', 'p', 'https://example.test')]
