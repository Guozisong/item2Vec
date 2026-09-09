# Training Progress and Summary Design

## Goal

Replace verbose training data dumps with concise Chinese runtime summaries and show terminal progress for the two existing long-running local loops.

## Scope

Pin the project's direct runtime and test dependencies to the versions validated in the current environment: `torch==2.14.0`, `pandas==2.3.3`, `numpy==2.2.6`, `transformers==5.16.1`, `gensim==4.4.0`, `scikit-learn==1.7.2`, `odps==4.0.0`, `pytest==9.1.1`, and `tqdm==4.70.0`. Do not add CUDA libraries, system packages, or other transitive dependencies to `requirements.txt`.

Wrap the per-batch loop in `generate_item_embedding` with a progress bar labeled `生成商品向量`, measured in batches. Wrap the per-order loop in `build_basket_indexes` with a progress bar labeled `构建训练购物篮`, measured in shopping baskets.

The wrappers cover existing iteration only; batching, filtering, model settings, device selection, and output formats remain unchanged.

## Training Output

`training.main` stops printing Python object types, vector dimensions, a vector row, and the complete basket list. It emits concise Chinese milestones instead:

- the number of raw behavior rows and distinct users after loading `order_item.csv`;
- the number of valid baskets after index construction and the number of source item vectors;
- the saved path of `trained_item.featCLS` after writing it.

Counts are derived from already-loaded data and are informational only; they do not affect filtering or training.

## Testing

Tests inject or monkeypatch the progress wrapper so no dynamic terminal output is required. They verify progress wrapping occurs for both loops, training emits the Chinese summary rather than raw vectors/baskets, and the existing artifact behavior remains unchanged. No ODPS request, model download, GPU execution, or real Word2Vec training occurs in tests.
