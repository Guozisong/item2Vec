# Production Recall Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a daily 30-day, order-level Item2Vec pipeline with order-independent co-occurrence training, confidence-aware text/behavior fusion, configurable recall modes, and validated atomic outputs.

**Architecture:** ODPS exports `order_id`, `prod_id`, and `dt`; training converts each order into a deduplicated, canonically sorted basket and trains SGNS with a fixed full-basket window. The behavior artifact carries vectors and per-product order counts. Inference selects a scenario preset, scales behavior weight by pair confidence, and writes mode-specific CSV output only after computation succeeds.

**Tech Stack:** Python 3.10+, pandas, NumPy, Gensim Word2Vec, pytest, Bash

---

### Task 1: Fetch and validate order-level training data

**Files:**
- Modify: `src/item2vec/data_fetch.py`
- Modify: `src/item2vec/training.py`
- Create: `tests/test_data_fetch.py`
- Modify: `tests/test_training.py`

- [ ] **Step 1: Write failing tests for the 30-day query and order baskets**

```python
# tests/test_data_fetch.py
from item2vec.data_fetch import ORDER_ITEM_SQL


def test_order_query_fetches_order_id_and_latest_thirty_days():
    normalized = " ".join(ORDER_ITEM_SQL.lower().split())
    assert "select order_id, user_id, prod_id, dt" in normalized
    assert "dateadd(getdate(), -29, 'dd')" in normalized
```

```python
# append to tests/test_training.py
import pandas as pd
import pytest


def test_prepare_order_baskets_groups_by_order_and_is_order_independent():
    rows = pd.DataFrame({
        "order_id": ["o1", "o1", "o1", "o2"],
        "prod_id": ["B", "A", "B", "A"],
        "dt": ["20260901"] * 4,
    })
    reversed_rows = rows.iloc[::-1].reset_index(drop=True)

    actual = training.prepare_order_baskets(rows, {"A": 0, "B": 1})
    reversed_actual = training.prepare_order_baskets(reversed_rows, {"A": 0, "B": 1})

    assert actual[0] == [["0", "1"]]
    assert reversed_actual[0] == [["0", "1"]]
    assert actual[1].tolist() == [2, 1]
    assert reversed_actual[1].tolist() == [2, 1]
    assert actual[2] == {"orders": 2, "valid_baskets": 1, "large_baskets": 0}
    assert reversed_actual[2] == actual[2]


def test_prepare_order_baskets_excludes_more_than_thirty_distinct_products():
    products = [f"P{i:02d}" for i in range(31)]
    rows = pd.DataFrame({
        "order_id": ["bulk"] * 31,
        "prod_id": products,
        "dt": ["20260901"] * 31,
    })

    baskets, counts, stats = training.prepare_order_baskets(
        rows, {product: index for index, product in enumerate(products)}
    )

    assert baskets == []
    assert counts.tolist() == [1] * 31
    assert stats == {"orders": 1, "valid_baskets": 0, "large_baskets": 1}


@pytest.mark.parametrize("missing", ["order_id", "prod_id", "dt"])
def test_prepare_order_baskets_requires_production_columns(missing):
    frame = pd.DataFrame({
        "order_id": ["o1"], "prod_id": ["A"], "dt": ["20260901"]
    }).drop(columns=missing)

    with pytest.raises(ValueError, match=missing):
        training.prepare_order_baskets(frame, {"A": 0})
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3.13 -m pytest -p no:cacheprovider -q tests/test_data_fetch.py tests/test_training.py
```

Expected: failures because the SQL lacks `order_id` and `prepare_order_baskets` does not exist.

- [ ] **Step 3: Implement order-level extraction and canonical basket preparation**

Use this query in `src/item2vec/data_fetch.py`:

```python
ORDER_ITEM_SQL = '''
select order_id, user_id, prod_id, dt
from unisrec_raw_data
where dt >= to_char(dateadd(getdate(), -29, 'dd'), 'yyyymmdd');
'''
```

Add this function to `src/item2vec/training.py`:

```python
def prepare_order_baskets(dataframe, item2index, max_basket_size=30):
    required = {"order_id", "prod_id", "dt"}
    missing = sorted(required - set(dataframe.columns))
    if missing:
        raise ValueError(f"订单数据缺少必需字段：{', '.join(missing)}")
    if dataframe.empty:
        raise ValueError("订单数据为空。")
    if dataframe[list(required)].isnull().any().any():
        raise ValueError("订单数据的 order_id、prod_id、dt 不允许为空。")
    if max_basket_size < 2:
        raise ValueError("max_basket_size must be at least 2")

    order_counts = np.zeros(len(item2index), dtype=np.int64)
    basket_indexes = []
    large_baskets = 0
    grouped = dataframe.groupby("order_id", sort=False)["prod_id"]
    for _, values in tqdm(grouped, desc="构建训练购物篮", unit="单", total=grouped.ngroups):
        known = {str(value) for value in values if str(value) in item2index}
        sequence = [str(item2index[item]) for item in known]
        sequence.sort(key=int)
        for index in sequence:
            order_counts[int(index)] += 1
        if len(sequence) > max_basket_size:
            large_baskets += 1
        elif len(sequence) >= 2:
            basket_indexes.append(sequence)

    stats = {
        "orders": grouped.ngroups,
        "valid_baskets": len(basket_indexes),
        "large_baskets": large_baskets,
    }
    return basket_indexes, order_counts, stats
```

- [ ] **Step 4: Run focused tests and verify GREEN**

Run the command from Step 2. Expected: all selected tests pass after updating old basket tests to the new order-level API.

- [ ] **Step 5: Commit the data preparation change**

```bash
git add src/item2vec/data_fetch.py src/item2vec/training.py tests/test_data_fetch.py tests/test_training.py
git commit -m "feat: build order-level training baskets"
```

### Task 2: Train full-basket behavior embeddings and publish metadata atomically

**Files:**
- Modify: `src/item2vec/training.py`
- Modify: `tests/test_training.py`

- [ ] **Step 1: Write failing tests for full-basket training and artifact metadata**

Add tests asserting:

```python
def test_train_uses_fixed_full_basket_window(monkeypatch):
    captured = {}

    class FakeVectors:
        index_to_key = ["0", "1"]

        def __getitem__(self, key):
            return np.ones(4, dtype=np.float32)

    class FakeWord2Vec:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.wv = FakeVectors()

        def build_vocab(self, sentences):
            self.sentences = sentences

        def train(self, sentences, **kwargs):
            return None

    monkeypatch.setitem(sys.modules, "gensim", types.ModuleType("gensim"))
    monkeypatch.setitem(
        sys.modules, "gensim.models", types.SimpleNamespace(Word2Vec=FakeWord2Vec)
    )

    training.train_item2vec(
        {"A": 0, "B": 1},
        [["0", "1"]] * 5,
        np.array([5, 5]),
        vector_size=4,
        max_basket_size=30,
        min_order_count=5,
    )

    assert captured["window"] == 30
    assert captured["shrink_windows"] is False
    assert captured["min_count"] == 1


def test_train_fails_when_no_behavior_pairs_survive():
    with pytest.raises(RuntimeError, match="行为共现数据不足"):
        training.train_item2vec(
            {"A": 0, "B": 1}, [["0", "1"]], np.array([1, 1]), min_order_count=5
        )


def test_write_behavior_embedding_stores_counts_metadata_and_replaces_atomically(tmp_path):
    existing = tmp_path / "behavior_item.npz"
    existing.write_bytes(b"old")
    output = training.write_behavior_embedding(
        np.eye(2),
        {"0": "A", "1": "B"},
        np.array([10, 20]),
        {"max_basket_size": 30, "min_order_count": 5},
        tmp_path,
    )

    assert output == existing
    with np.load(output, allow_pickle=False) as artifact:
        assert artifact["order_counts"].tolist() == [10, 20]
        assert json.loads(artifact["metadata"].item()) == {
            "max_basket_size": 30, "min_order_count": 5
        }
```

- [ ] **Step 2: Run tests and verify RED**

```bash
PYTHONDONTWRITEBYTECODE=1 python3.13 -m pytest -p no:cacheprovider -q tests/test_training.py
```

Expected: signature, fixed-window, failure-mode, and metadata assertions fail.

- [ ] **Step 3: Implement explicit eligibility, fixed windows, and atomic NPZ output**

Change `train_item2vec` to accept prepared index baskets and order counts. Filter each basket to products meeting `min_order_count`, discard filtered baskets shorter than two, and raise `RuntimeError` when none remain. Initialize Word2Vec with:

```python
model = Word2Vec(
    vector_size=vector_size,
    window=max_basket_size,
    min_count=1,
    sg=1,
    negative=negative,
    sample=1e-4,
    workers=workers,
    epochs=epochs,
    alpha=0.002,
    min_alpha=0.0005,
    shrink_windows=False,
)
```

Write and validate the artifact through a same-directory temporary file:

```python
def write_behavior_embedding(embedding, index2item, order_counts, metadata, downstream_dir):
    output_path = Path(downstream_dir) / "behavior_item.npz"
    temporary_path = output_path.with_suffix(".npz.tmp")
    item_ids = np.asarray([str(index2item[str(i)]) for i in range(len(index2item))])
    try:
        with temporary_path.open("wb") as stream:
            np.savez(
                stream,
                vectors=np.asarray(embedding, dtype=np.float32),
                item_ids=item_ids,
                order_counts=np.asarray(order_counts, dtype=np.int64),
                metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
            )
        with np.load(temporary_path, allow_pickle=False) as artifact:
            if artifact["vectors"].shape[0] != len(item_ids):
                raise ValueError("行为向量行数与商品 ID 不匹配。")
            if not np.isfinite(artifact["vectors"]).all():
                raise ValueError("行为向量包含非有限值。")
        os.replace(temporary_path, output_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return output_path
```

Update `main` to call `prepare_order_baskets`, pass counts into training and writing, and log orders, valid baskets, large baskets, and covered products.

- [ ] **Step 4: Run focused and full tests**

```bash
PYTHONDONTWRITEBYTECODE=1 python3.13 -m pytest -p no:cacheprovider -q tests/test_training.py
PYTHONDONTWRITEBYTECODE=1 python3.13 -m pytest -p no:cacheprovider -q
```

Expected: focused tests pass; full suite passes after updating callers for the new artifact signature.

- [ ] **Step 5: Commit behavior training**

```bash
git add src/item2vec/training.py tests/test_training.py tests/test_score_fusion.py tests/test_inference.py
git commit -m "feat: train full-basket behavior embeddings"
```

### Task 3: Add recall modes and confidence-aware fusion

**Files:**
- Modify: `src/item2vec/inference.py`
- Modify: `tests/test_score_fusion.py`
- Modify: `tests/test_inference.py`

- [ ] **Step 1: Write failing mode and confidence tests**

```python
@pytest.mark.parametrize(
    ("mode", "expected"),
    [("similar", .85), ("complement", .20), ("hybrid", .60)],
)
def test_resolve_text_weight_uses_mode_presets(mode, expected):
    assert inference.resolve_text_weight(mode, None) == expected


def test_resolve_text_weight_allows_override_and_rejects_unknown_mode():
    assert inference.resolve_text_weight("hybrid", .72) == .72
    with pytest.raises(ValueError, match="recall_mode"):
        inference.resolve_text_weight("unknown", None)


def test_behavior_weight_is_scaled_by_lower_order_confidence():
    text = np.array([[1., 0.], [1., 0.]])
    behavior = np.array([[1., 0.], [0., 1.]])
    result = inference.rank_items(
        text,
        {"0": "A", "1": "B"},
        [0],
        1,
        behavior_vectors=behavior,
        behavior_order_counts=np.array([50, 5]),
        text_weight=.2,
        full_confidence_orders=50,
    )
    assert result.iloc[0].similarity == pytest.approx(.92)


def test_zero_behavior_falls_back_completely_to_text_even_with_high_count():
    text = np.array([[1., 0.], [.8, .6]])
    behavior = np.array([[1., 0.], [0., 0.]])
    result = inference.rank_items(
        text,
        {"0": "A", "1": "B"},
        [0],
        1,
        behavior_vectors=behavior,
        behavior_order_counts=np.array([100, 100]),
        text_weight=.2,
    )
    assert result.iloc[0].similarity == pytest.approx(.8)
```

- [ ] **Step 2: Run tests and verify RED**

```bash
PYTHONDONTWRITEBYTECODE=1 python3.13 -m pytest -p no:cacheprovider -q tests/test_score_fusion.py tests/test_inference.py
```

Expected: failures for missing mode resolver, missing counts, and the old constant-weight calculation.

- [ ] **Step 3: Implement presets and pair confidence**

Add constants and validation:

```python
MODE_TEXT_WEIGHTS = {"similar": .85, "complement": .20, "hybrid": .60}


def resolve_text_weight(recall_mode, override):
    if recall_mode not in MODE_TEXT_WEIGHTS:
        raise ValueError(f"Unknown recall_mode: {recall_mode}")
    weight = MODE_TEXT_WEIGHTS[recall_mode] if override is None else override
    if not np.isfinite(weight) or not 0 <= weight <= 1:
        raise ValueError("text_weight must be finite and between 0 and 1")
    return float(weight)
```

Load `order_counts` from `behavior_item.npz`, validate its length and non-negative values, and return it with the vectors. In each inference block calculate:

```python
confidence = np.minimum(behavior_order_counts / full_confidence_orders, 1.0)
pair_confidence = np.minimum(confidence[block_sources, None], confidence[None, :])
pair_available = available[block_sources, None] & available[None, :]
effective_behavior_weight = (1 - text_weight) * pair_confidence * pair_available
similarities = (
    (1 - effective_behavior_weight) * text_scores
    + effective_behavior_weight * behavior_scores
)
```

Require `full_confidence_orders` to be a positive integer. Update query/export functions and CLI parsers to accept `--recall-mode`, optional `--text-weight`, and `--full-confidence-orders`.

- [ ] **Step 4: Run focused and full tests**

Run the command from Step 2, then the complete pytest command from Task 2. Expected: all tests pass.

- [ ] **Step 5: Commit scoring changes**

```bash
git add src/item2vec/inference.py tests/test_score_fusion.py tests/test_inference.py
git commit -m "feat: add confidence-aware recall modes"
```

### Task 4: Publish mode-specific CSV files atomically

**Files:**
- Modify: `src/item2vec/inference.py`
- Modify: `tests/test_inference.py`

- [ ] **Step 1: Write failing output safety tests**

```python
def test_export_uses_mode_specific_filename(tmp_path):
    vectors = np.eye(2, 768, dtype=np.float32)
    _write_artifacts(tmp_path, ["A", "B"], vectors, order_counts=[50, 50])

    output = inference.export_all(tmp_path, top_k=1, recall_mode="complement")

    assert output == tmp_path / "item_similarity_complement.csv"


def test_atomic_csv_write_preserves_previous_file_on_failure(monkeypatch, tmp_path):
    output = tmp_path / "item_similarity_hybrid.csv"
    output.write_text("previous\n", encoding="utf-8")
    frame = pd.DataFrame({"value": [1]})

    def fail(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(frame, "to_csv", fail)
    with pytest.raises(OSError, match="disk full"):
        inference.write_csv_atomic(frame, output)

    assert output.read_text(encoding="utf-8") == "previous\n"
```

- [ ] **Step 2: Run the focused test and verify RED**

```bash
PYTHONDONTWRITEBYTECODE=1 python3.13 -m pytest -p no:cacheprovider -q tests/test_inference.py
```

Expected: mode-specific filename and `write_csv_atomic` tests fail.

- [ ] **Step 3: Implement same-directory atomic CSV replacement**

```python
def write_csv_atomic(dataframe, output_path):
    output_path = Path(output_path)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    try:
        dataframe.to_csv(temporary_path, index=False)
        if not temporary_path.is_file() or temporary_path.stat().st_size == 0:
            raise ValueError("相似度 CSV 输出为空。")
        temporary_path.replace(output_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return output_path
```

Use it in query and export. Export to `item_similarity_<recall_mode>.csv`; keep `query_<ITEM_ID>.csv` for backward-compatible single-item queries.

- [ ] **Step 4: Run focused and full tests**

Run the focused command from Step 2 and the full pytest command from Task 2. Expected: all tests pass and previous files survive simulated write failure.

- [ ] **Step 5: Commit atomic output handling**

```bash
git add src/item2vec/inference.py tests/test_inference.py
git commit -m "feat: publish recall outputs atomically"
```

### Task 5: Expose production settings through Bash

**Files:**
- Modify: `scripts/train.sh`
- Modify: `scripts/query_similar.sh`
- Modify: `scripts/export_similarities.sh`
- Modify: `tests/test_scripts.py`

- [ ] **Step 1: Write failing Bash forwarding tests**

Update the script tests to require these defaults:

```python
def test_train_script_forwards_production_defaults(tmp_path):
    result, arguments = _run_train_script_with_stub(tmp_path, [])
    assert result.returncode == 0
    assert arguments == [
        str(tmp_path / "dataset" / "raw"),
        str(tmp_path / "dataset" / "downstream"),
        "--vector-size", "128",
        "--max-basket-size", "30",
        "--negative", "15",
        "--epochs", "10",
        "--min-order-count", "5",
    ]


def test_export_script_forwards_mode_and_confidence_defaults(tmp_path):
    result, arguments = _run_inference_script_with_stub(
        tmp_path, "export_similarities.sh", []
    )
    assert result.returncode == 0
    assert arguments == [
        "export",
        str(tmp_path / "dataset" / "downstream"),
        "--top-k", "10",
        "--block-size", "512",
        "--recall-mode", "hybrid",
        "--full-confidence-orders", "50",
    ]
```

Add tests for `RECALL_MODE=complement`, `TEXT_WEIGHT=.3`, and `FULL_CONFIDENCE_ORDERS=80` environment overrides, plus invalid argument counts.

- [ ] **Step 2: Run script tests and verify RED**

```bash
PYTHONDONTWRITEBYTECODE=1 python3.13 -m pytest -p no:cacheprovider -q tests/test_scripts.py
```

Expected: assertions fail because existing scripts still pass window and constant text weight arguments.

- [ ] **Step 3: Implement Bash defaults and overrides**

In `scripts/train.sh`, keep the existing four-position interface but redefine its second value as `MAX_BASKET_SIZE`:

```bash
vector_size="${1:-128}"
max_basket_size="${2:-30}"
negative="${3:-15}"
epochs="${4:-10}"
min_order_count="${MIN_ORDER_COUNT:-5}"
```

Pass `--max-basket-size` and `--min-order-count`; remove `--window`.

In query/export scripts resolve:

```bash
recall_mode="${RECALL_MODE:-hybrid}"
full_confidence_orders="${FULL_CONFIDENCE_ORDERS:-50}"
text_weight="${TEXT_WEIGHT:-}"
```

Pass mode and confidence arguments every time. Append `--text-weight "${text_weight}"` only when the override is non-empty. Preserve strict shell error handling and artifact checks.

- [ ] **Step 4: Run script and syntax checks**

```bash
PYTHONDONTWRITEBYTECODE=1 python3.13 -m pytest -p no:cacheprovider -q tests/test_scripts.py
bash -n scripts/*.sh
```

Expected: script tests pass and Bash reports no syntax errors.

- [ ] **Step 5: Commit Bash configuration**

```bash
git add scripts/train.sh scripts/query_similar.sh scripts/export_similarities.sh tests/test_scripts.py
git commit -m "feat: configure production recall scripts"
```

### Task 6: Update operator documentation and run release verification

**Files:**
- Modify: `README.md`
- Test: entire repository

- [ ] **Step 1: Update README commands, artifacts, and scoring formula**

Document the required `order_id` field, latest-30-day query, per-order baskets, 30-product anomaly cutoff, five-order eligibility threshold, three recall modes, optional environment overrides, confidence formula, atomic behavior artifact, and mode-specific filenames. Use these examples:

```bash
bash scripts/train.sh
RECALL_MODE=similar bash scripts/export_similarities.sh 20 512
RECALL_MODE=complement FULL_CONFIDENCE_ORDERS=80 bash scripts/export_similarities.sh
RECALL_MODE=hybrid TEXT_WEIGHT=0.7 bash scripts/query_similar.sh ITEM_ID 10
```

- [ ] **Step 2: Run complete verification**

```bash
PYTHONDONTWRITEBYTECODE=1 python3.13 -m pytest -p no:cacheprovider -q
bash -n scripts/*.sh
git diff --check
```

Expected: all tests pass, every shell script parses, and no whitespace errors are reported.

- [ ] **Step 3: Inspect the scoped diff and working tree**

```bash
git diff --stat 1d19815..HEAD
git status --short
```

Expected: only the production recall source, tests, scripts, README, spec, and plan are tracked changes; the pre-existing unrelated plan files remain untouched.

- [ ] **Step 4: Commit documentation**

```bash
git add README.md
git commit -m "docs: explain production recall workflow"
```

- [ ] **Step 5: Record final evidence**

Run:

```bash
git log --oneline --decorate -8
git status --short --branch
```

Expected: the production recall commits are present on `feature/score-fusion`; no implementation files remain modified, and only pre-existing unrelated untracked files may remain.
