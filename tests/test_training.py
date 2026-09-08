from item2vec.training import build_basket_indexes


def test_build_basket_indexes_drops_unknown_and_invalid_lengths():
    baskets = [["A", "missing", "B"], ["A"], list("ABCDEFGHIJKLMNOPQRSTU")]
    item2index = {code: index for index, code in enumerate("ABCDEFGHIJKLMNOPQRSTU")}
    assert build_basket_indexes(baskets, item2index) == [["0", "1"]]
