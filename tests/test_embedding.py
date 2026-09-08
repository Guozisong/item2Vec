import json

from item2vec.embedding import build_item_index


def test_build_item_index_preserves_numeric_ids_and_reloads_string_keys(tmp_path):
    item_csv = tmp_path / "item.csv"
    item_csv.write_text("prod_id,prod_description\n101,first\n202,second\n", encoding="utf-8")

    item2index = build_item_index(item_csv, tmp_path)

    with open(tmp_path / "index2item.json", encoding="utf-8") as file:
        assert json.load(file) == {"0": 101, "1": 202}
    assert item2index == {"101": 0, "202": 1}
