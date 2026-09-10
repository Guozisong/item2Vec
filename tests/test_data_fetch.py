from item2vec.data_fetch import ORDER_ITEM_SQL


def test_order_item_sql_selects_order_level_items_from_last_30_days():
    normalized_sql = " ".join(ORDER_ITEM_SQL.split()).lower()

    assert "select order_id, user_id, prod_id, dt from unisrec_raw_data" in normalized_sql
    assert "dt >= to_char(dateadd(getdate(), -29, 'dd'), 'yyyymmdd')" in normalized_sql
