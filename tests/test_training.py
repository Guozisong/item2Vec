import numpy as np

from item2vec.training import build_basket_indexes, write_trained_embedding


def test_build_basket_indexes_drops_unknown_and_invalid_lengths():
    baskets = [["A", "missing", "B"], ["A"], list("ABCDEFGHIJKLMNOPQRSTU")]
    item2index = {code: index for index, code in enumerate("ABCDEFGHIJKLMNOPQRSTU")}
    assert build_basket_indexes(baskets, item2index) == [["0", "1"]]


def test_write_trained_embedding_writes_float32_artifact_without_similarity_csv(tmp_path):
    embedding = np.array([[1.5, 2.5], [3.5, 4.5]], dtype=np.float64)

    output_path = write_trained_embedding(embedding, tmp_path)

    assert output_path == tmp_path / "trained_item.featCLS"
    assert np.array_equal(np.fromfile(output_path, dtype=np.float32), embedding.astype(np.float32).ravel())
    assert not (tmp_path / "item_cosine_similarity.csv").exists()
