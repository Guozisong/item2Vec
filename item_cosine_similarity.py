import os
import numpy as np
import json
import pandas as pd
from odps import ODPS
from odps.models import Schema, Column
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
from item2Embedding import load_plm_embedding, load_index_item

def write_to_odps(df1, table_name1):
    # 建立链接。
    load_dotenv("/ml/output/.env")
    access_id = os.getenv("ALI_ACCESS_ID")
    access_key = os.getenv("ALI_SECRET_ACCESS_KEY")
    project = 'jxaidataworks'
    endpoint = 'https://service.cn-hangzhou-vpc.maxcompute.aliyun-inc.com/api'

    # 初始化ODPS对象
    odps = ODPS(access_id, access_key, project, endpoint)


    columns = [
        Column(name='master_prod_id', type='string', comment='主商品'),
        Column(name='slave_prod_id', type='string', comment='从商品'),
        Column(name='sim_score', type='double', comment='相似分数')]

    schema = Schema(columns=columns)

    # 检查表是否存在，如果存在则删除
    if odps.exist_table(table_name1):
        odps.delete_table(table_name1, if_exists=True)
        print(f"Table {table_name1} has been dropped.")

    odps.create_table(table_name1, schema, if_not_exists=True)
    print(f"Table {table_name1} already exists.")

    table = odps.get_table(table_name1)

    # 将DataFrame写入ODPS表
    odps.write_table(table_name1, df1, overwrite=True)

    print(f"Data has been written to {table_name1}")


def get_cosine_similarity(itemEmbedding, index2Item, topk, valid_items_list, filter_target_items_list=None):
    sim_matrix = cosine_similarity(itemEmbedding)
    records = []

    # 过滤集合
    if filter_target_items_list is not None:
        filter_set = set(map(str, filter_target_items_list))
    else:
        filter_set = set()

    for i in range(sim_matrix.shape[0]):
        item_id = str(index2Item[str(i)])
        # master_prod为当日上架商品
        if item_id not in valid_items_list:
            continue

        # 相似度从大到小
        top_indices = np.argsort(-sim_matrix[i])

        cnt = 0
        for j in top_indices:
            if j == i:
                continue

            sim_item_id = str(index2Item[str(j)])

            # slave_prod为当日上架商品
            if sim_item_id not in valid_items_list:
                continue
            
            # slave_prod 规则过滤
            if sim_item_id in filter_set:
                continue

            sim_score = float(sim_matrix[i, j])
            records.append((item_id, sim_item_id, sim_score))

            cnt += 1
            if cnt >= topk:
                break

    return pd.DataFrame(records, columns=["master_prod_id", "slave_prod_id", "sim_score"])



if __name__ == '__main__':
    topk = 300
    raw_path = '/ml/output/dataset/raw/'
    feat_path = '/ml/output/dataset/downstream/'
    
    itemEmbedding = load_plm_embedding(feat_path, "lianhua_item.feat1CLS")
    trained_itemEmbedding = load_plm_embedding(feat_path, "trained_linhua_item.featCLS")
    index2Item = load_index_item(feat_path)

    delta = np.linalg.norm(trained_itemEmbedding - itemEmbedding, axis=1)
    print("mean delta:", delta.mean())
    print("median delta:", np.median(delta))

    
    filter_target_items_list = pd.read_csv(os.path.join(raw_path, 'filter_target_items.csv'))["prod_id"].astype(str).tolist()
    valid_items_list = pd.read_csv(os.path.join(raw_path, 'valid_items.csv'))["prod_id"].astype(str).tolist()

    
    item_similarity_with_itemEmbedding = get_cosine_similarity(itemEmbedding, index2Item, topk, valid_items_list, filter_target_items_list)
    item_similarity_with_trained_itemEmbedding = get_cosine_similarity(trained_itemEmbedding, index2Item, topk, valid_items_list, filter_target_items_list)
   
    write_to_odps(item_similarity_with_itemEmbedding, "prod_similarty_with_bert_t")
    write_to_odps(item_similarity_with_trained_itemEmbedding, "prod_similarty_with_bert_item2Vec_t")
