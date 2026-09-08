import os


DEFAULT_ENDPOINT = 'https://service.cn-hangzhou-vpc.maxcompute.aliyun-inc.com/api'


ITEM_SQL = '''
select prod_id, prod_description
from unisrec_items_info
where prod_id in (
    select prod_id from lh_rec_gds_station_base_pool_tmp group by prod_id
)
group by prod_id, prod_description;
'''

ORDER_ITEM_SQL = 'select user_id, prod_id, dt from unisrec_raw_data;'


def fetch_data(output_dir, access_id, access_key):
    from odps import ODPS

    odps = ODPS(
        access_id,
        access_key,
        os.environ.get('ALI_PROJECT', ''),
        os.environ.get('ALI_ENDPOINT') or DEFAULT_ENDPOINT,
    )
    for sql, filename in ((ITEM_SQL, 'item.csv'), (ORDER_ITEM_SQL, 'order_item.csv')):
        dataframe = odps.execute_sql(sql).open_reader(tunnel=True).to_pandas(n_process=4)
        dataframe.to_csv(os.path.join(output_dir, filename), index=False, encoding='utf-8')


def main():
    required_names = ('ALI_ACCESS_ID', 'ALI_SECRET_ACCESS_KEY', 'ALI_PROJECT')
    if any(not os.environ.get(name) for name in required_names):
        raise RuntimeError('Missing ODPS credentials')

    output_dir = os.path.join(os.getcwd(), 'dataset', 'raw')
    fetch_data(output_dir, os.environ['ALI_ACCESS_ID'], os.environ['ALI_SECRET_ACCESS_KEY'])


if __name__ == '__main__':
    main()
