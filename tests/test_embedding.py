import json
import sys
from types import SimpleNamespace

import numpy as np

from item2vec import embedding
from item2vec.embedding import build_item_index


def test_build_item_index_preserves_numeric_ids_and_reloads_string_keys(tmp_path):
    item_csv = tmp_path / "item.csv"
    item_csv.write_text("prod_id,prod_description\n101,first\n202,second\n", encoding="utf-8")

    item2index = build_item_index(item_csv, tmp_path)

    with open(tmp_path / "index2item.json", encoding="utf-8") as file:
        assert json.load(file) == {"0": 101, "1": 202}
    assert item2index == {"101": 0, "202": 1}


def test_generate_item_embedding_shows_batch_progress_and_writes_all_rows(tmp_path, monkeypatch, capsys):
    progress_calls = []

    def fake_tqdm(iterable, **kwargs):
        progress_calls.append((list(iterable), kwargs))
        return iter(progress_calls[-1][0])

    class TokenBatch(dict):
        def to(self, device):
            return self

    class Tokenizer:
        def __init__(self):
            self.batches = []

        def __call__(self, sentences, **kwargs):
            self.batches.append(sentences)
            return TokenBatch()

    class Model:
        def __init__(self):
            self.next_value = 0

        def __call__(self, **encoded_sentences):
            batch_size = len(tokenizer.batches[-1])
            values = FakeTensor(np.arange(self.next_value, self.next_value + batch_size * 2,
                                          dtype=np.float32).reshape(batch_size, 1, 2))
            self.next_value += batch_size * 2
            return SimpleNamespace(last_hidden_state=values)

    class FakeTensor:
        def __init__(self, values):
            self.values = values

        def __getitem__(self, index):
            return FakeTensor(self.values[index])

        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self.values

    class FakeTorch:
        @staticmethod
        def cat(tensors, dim):
            return FakeTensor(np.concatenate([tensor.values for tensor in tensors], axis=dim))

    tokenizer = Tokenizer()
    model = Model()
    monkeypatch.setitem(sys.modules, "torch", FakeTorch())
    monkeypatch.setattr(embedding, "tqdm", fake_tqdm)

    embedding.generate_item_embedding(
        word_drop_ratio=-1,
        emb_type="CLS",
        device="cpu",
        output_path=tmp_path,
        item2index={str(index): index for index in range(5)},
        item_text_list=[[str(index), f"item {index}"] for index in range(5)],
        plm_tokenizer=tokenizer,
        plm_model=model,
    )

    assert tokenizer.batches == [["item 0", "item 1", "item 2", "item 3"], ["item 4"]]
    assert progress_calls == [
        ([0, 4], {"total": 2, "desc": "生成商品向量", "unit": "批"}),
    ]
    assert np.fromfile(tmp_path / "item.feat1CLS", dtype=np.float32).reshape(5, 2).tolist() == [
        [0.0, 1.0], [2.0, 3.0], [4.0, 5.0], [6.0, 7.0], [8.0, 9.0],
    ]
    assert "商品向量生成完成：共 5 个商品，已保存至" in capsys.readouterr().out
