from item2vec.data_fetch import ORDER_ITEM_SQL


def test_order_item_sql_selects_order_level_items_from_last_30_days():
    normalized_sql = " ".join(ORDER_ITEM_SQL.split()).lower()

    assert "select order_id, user_id, prod_id, dt from unisrec_raw_data" in normalized_sql
    assert "dt >= to_char(dateadd(getdate(), -29, 'dd'), 'yyyymmdd')" in normalized_sql


def test_main_creates_and_forwards_explicit_output_directory(monkeypatch, tmp_path):
    monkeypatch.setenv("ALI_ACCESS_ID", "id")
    monkeypatch.setenv("ALI_SECRET_ACCESS_KEY", "key")
    monkeypatch.setenv("ALI_PROJECT", "project")
    captured = {}

    def fake_fetch_data(output_dir, access_id, access_key):
        captured.update(
            output_dir=output_dir, access_id=access_id, access_key=access_key
        )

    monkeypatch.setattr("item2vec.data_fetch.fetch_data", fake_fetch_data)
    output_dir = tmp_path / "raw"

    from item2vec import data_fetch
    data_fetch.main([str(output_dir)])

    assert output_dir.is_dir()
    assert captured == {
        "output_dir": str(output_dir),
        "access_id": "id",
        "access_key": "key",
    }
