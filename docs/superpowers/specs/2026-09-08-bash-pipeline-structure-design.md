# Bash Pipeline Structure Design

## Goal

Reorganize the Item2Vec pipeline into an importable Python package and make
Bash scripts the supported way to fetch data, generate embeddings, train, and
run the end-to-end pipeline. Existing root-level Python commands will be
removed rather than preserved as compatibility entry points.

## Architecture

`src/item2vec/` will contain only reusable Python logic. `embedding.py` will
create M3E embeddings and item mappings; `training.py` will build baskets,
train the initialized Word2Vec model, and export results; `io.py` will contain
mapping, embedding, model, and similarity helpers. `data_fetch.py` will own the
ODPS download operation without embedding credentials.

`scripts/` will contain executable Bash entry points. `fetch_data.sh`,
`generate_embeddings.sh`, and `train.sh` will each validate required paths and
invoke a Python module. `run_pipeline.sh` will call them in dependency order.
Every script will resolve the repository root from its own location, use
`set -euo pipefail`, and accept no credentials as arguments. ODPS credentials
remain in the untracked `dataset/raw/.env` environment file.

## Data Flow

`fetch_data.sh` writes `dataset/raw/item.csv` and `dataset/raw/order_item.csv`.
`generate_embeddings.sh` reads `item.csv` plus `dataset/m3e-base/`, then writes
the index mappings and `item.feat1CLS` under `dataset/downstream/`.
`train.sh` consumes those outputs with `order_item.csv`, then writes
`trained_item.featCLS` and `item_cosine_similarity.csv`. The orchestration
script simply chains those stages and fails at the first unsuccessful stage.

## Quality and Scope

Add `pytest` tests for pure mapping, basket construction, and path validation
without requiring ODPS access, CUDA, downloaded model weights, or production
data. Preserve existing algorithm parameters, file formats, output names, and
the current embedding dimension. Do not add a CLI framework, new model
features, or data transformations.
