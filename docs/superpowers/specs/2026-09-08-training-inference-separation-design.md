# Training and Inference Separation Design

## Goal

Separate offline Item2Vec training from similarity inference. Training produces
the fused embedding artifact only; independent Bash entry points query one item
or export all item similarities as CSV using the trained vectors.

## Architecture

`src/item2vec/training.py` will stop importing similarity helpers and will only
write `dataset/downstream/trained_item.featCLS`. A new
`src/item2vec/inference.py` module will load that artifact together with
`index2item.json`, validate their dimensions, and provide exact cosine-similarity
operations.

`scripts/query_similar.sh <item-id> [top-k]` will write
`dataset/downstream/query_<safe-item-id>.csv`. `scripts/export_similarities.sh
[top-k]` will write `dataset/downstream/item_cosine_similarity.csv`. Both use a
default Top-K of 10 and return the columns `master_prod_id`, `slave_prod_id`, and
`similarity`.

## Computation and Data Flow

Single-item inference compares one trained vector with the complete embedding
matrix. Batch export processes source items in fixed-size blocks so catalogs of
10,000 to 100,000 items do not require an in-memory `N x N` matrix. Results are
exact, exclude the source item, and use deterministic descending similarity
order. No approximate-index dependency is introduced.

`scripts/run_pipeline.sh` continues to end after training. Inference is invoked
separately, so repeated queries or exports never retrain the model, fetch ODPS
data, or run M3E encoding.

## Validation and Failure Handling

Inference fails clearly for missing mappings or embeddings, inconsistent item
counts, invalid Top-K values, and unknown item IDs. Item IDs are sanitized only
for output filenames; CSV values retain their original identifiers.

Tests use small NumPy arrays and temporary directories to verify ranking,
self-exclusion, CSV schemas, batch-block equivalence, filename safety, invalid
inputs, and Bash entry points. They require no ODPS access, model weights, GPU,
or production data.

## Scope

Preserve existing training parameters and binary embedding format. Restore the
original automatic CUDA/CPU selection; do not include the discarded CPU-only
working-tree change. Do not add FAISS, an online service, or API endpoints.
