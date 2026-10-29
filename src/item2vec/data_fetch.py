"""Prepare catalog and order CSVs from explicit CSV, ODPS table, or SQL sources."""

import argparse
import os
import re
import tempfile
from pathlib import Path

import pandas as pd


DEFAULT_ENDPOINT = 'https://service.cn-hangzhou-vpc.maxcompute.aliyun-inc.com/api'
FIELDS = {'item': ('prod_id', 'prod_description'), 'order': ('order_id', 'prod_id', 'dt')}


def source_sql(table=None, query_file=None):
    if query_file is not None:
        sql = Path(query_file).read_text(encoding='utf-8-sig').strip()
        if not sql:
            raise ValueError('SQL query file must not be empty')
        return sql
    if not table or not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*){0,2}', table):
        raise ValueError('Invalid ODPS table name')
    return f'SELECT * FROM {table}'


def connect_odps():
    settings = {name: os.environ.get(upper) or os.environ.get(name)
                for name, upper in (('access_id', 'ALI_ACCESS_ID'),
                                    ('access_key', 'ALI_SECRET_ACCESS_KEY'),
                                    ('project', 'ALI_PROJECT'), ('endpoint', 'ALI_ENDPOINT'))}
    if any(not settings[name] for name in ('access_id', 'access_key', 'project')):
        raise RuntimeError('Missing ODPS credentials: access_id, access_key, project')
    from odps import ODPS

    return ODPS(settings['access_id'], settings['access_key'], settings['project'],
                settings['endpoint'] or DEFAULT_ENDPOINT)


def normalize(dataframe, kind):
    required = FIELDS[kind]
    missing = [name for name in required if name not in dataframe.columns]
    if missing:
        raise ValueError(f'{kind} source requires columns: {", ".join(missing)}')
    if dataframe.empty:
        raise ValueError(f'{kind} source must not be empty')
    for column in required:
        if dataframe[column].isna().any() or dataframe[column].astype(str).str.strip().eq('').any():
            raise ValueError(f'{kind} source contains blank {column}')
    fields = list(required)
    if kind == 'order' and 'user_id' in dataframe.columns:
        fields = ['order_id', 'user_id', 'prod_id', 'dt']
    return dataframe.loc[:, fields].fillna('').astype(str)


def fetch_data(output_dir, sources):
    # Resolve local files and validate query/table inputs before contacting ODPS.
    prepared = {}
    for kind, (mode, value) in sources.items():
        prepared[kind] = (pd.read_csv(value, dtype=str, keep_default_na=False, encoding='utf-8-sig')
                          if mode == 'csv' else source_sql(
                              table=value if mode == 'table' else None,
                              query_file=value if mode == 'query-file' else None))
    client = connect_odps() if any(mode != 'csv' for mode, _ in sources.values()) else None
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output_dir, prefix='.fetch-') as temporary:
        staged = []
        for kind in ('item', 'order'):
            source = prepared[kind]
            if isinstance(source, str):
                source = client.execute_sql(source).open_reader(tunnel=True).to_pandas(n_process=4)
            dataframe = normalize(source, kind)
            filename = 'item.csv' if kind == 'item' else 'order_item.csv'
            path = Path(temporary) / filename
            dataframe.to_csv(path, index=False, encoding='utf-8')
            staged.append((path, output_dir / filename))
            print(f'{kind}: {len(dataframe)} rows, source={sources[kind][0]}, output={output_dir / filename}')
        for source, destination in staged:
            os.replace(source, destination)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output_dir')
    for kind in FIELDS:
        group = parser.add_mutually_exclusive_group(required=True)
        for mode in ('csv', 'table', 'query-file'):
            group.add_argument(f'--{kind}-input-{mode}')
    args = parser.parse_args(argv)
    sources = {}
    for kind in FIELDS:
        for mode in ('csv', 'table', 'query-file'):
            value = getattr(args, f'{kind}_input_{mode.replace("-", "_")}')
            if value is not None:
                sources[kind] = (mode, value)
    fetch_data(args.output_dir, sources)


if __name__ == '__main__':
    main()
