import os
from odps import ODPS


def get_df_from_odps(sql, saved_path, save_name):
    access_id = ''
    access_key = ''
    project = ''
    endpoint = 'https://service.cn-hangzhou-vpc.maxcompute.aliyun-inc.com/api'

    # 初始化ODPS对象
    odps = ODPS(access_id, access_key, project, endpoint)

    # 读取商品属性数据
    query_job = odps.execute_sql(sql)
    prod_attr_df = query_job.open_reader(tunnel=True)
    prod_attr_df = prod_attr_df.to_pandas(n_process=4)
    csv_file_path = os.path.join(saved_path, save_name)
    prod_attr_df.to_csv(csv_file_path, index=False, encoding='utf-8')


if __name__ == '__main__':
    sql1 = '''
              select prod_id, prod_description 
              from unisrec_items_info
              where prod_id in 
              (
                select prod_id from lh_rec_gds_station_base_pool_tmp group by prod_id
              )
              group by prod_id, prod_description;'''

    sql2 = '''select user_id, prod_id, dt from unisrec_raw_data;'''

    saved_path = "./"
    get_df_from_odps(sql1, saved_path, "lianhua_item.csv")
    get_df_from_odps(sql2, saved_path, "lianhua_order_item.csv")