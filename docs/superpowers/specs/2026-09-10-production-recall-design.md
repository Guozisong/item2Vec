# Production Recall Design

## Goal

Build a daily offline item-recall pipeline for 20,000–30,000 products and 20,000–30,000 orders per day. Each run retrains from the latest 30 days of orders and produces configurable similarity CSV files. The design prioritizes production relevance and cold-start coverage rather than strict reproduction of the Item2Vec paper.

## Training Data

Input orders must contain `order_id`, `prod_id`, and `dt`. Build one basket per `order_id`, deduplicate products within the order, and treat the basket as an unordered set. Single-product orders do not create co-occurrence pairs but still contribute to product order counts.

Exclude baskets containing more than 30 distinct products as anomalous bulk orders. Log the number excluded. Generate every directed product pair within each retained basket, so training is independent of CSV row order. Products must occur in at least five orders to receive behavior embeddings.

Retrain behavior embeddings from scratch each day. Store vectors, ordered product IDs, product order counts, and training metadata in `behavior_item.npz`.

## Scoring

Normalize text and behavior vectors separately. Configure scoring with `RECALL_MODE`:

- `similar`: text 0.85, behavior 0.15
- `complement`: text 0.20, behavior 0.80
- `hybrid`: text 0.60, behavior 0.40; default mode

Allow `TEXT_WEIGHT` to override the selected preset. Behavior confidence is `min(order_count / 50, 1)`. For a product pair, use the lower confidence of the two products. The effective behavior weight is the preset behavior weight multiplied by pair confidence. Apply the remaining weight to text similarity. Missing behavior on either product therefore falls back completely to text.

The full-confidence threshold defaults to 50 orders and is configurable through the Bash scripts.

## Outputs and Runtime

Use exact blockwise cosine similarity for the current catalog size; do not add an ANN dependency. Name batch output for the selected mode as `item_similarity_<mode>.csv`. Preserve the existing query and CSV schemas.

Write artifacts to temporary files, validate them, and atomically replace production outputs only after validation succeeds. Required checks include non-empty inputs, required columns, contiguous product mappings, matching artifact IDs, finite vectors, valid dimensions, and non-zero behavior coverage. A failed run must preserve the previous valid artifacts.

Logs must report input orders, valid baskets, excluded large baskets, behavior-covered products, selected mode and weights, progress, and output row counts.

## Verification

Tests must cover order-level grouping, deduplication, row-order independence, the 30-product cutoff, full pair coverage, behavior confidence, mode defaults, manual weight overrides, text fallback, artifact validation, and Bash argument propagation. Run the full Python test suite and `bash -n scripts/*.sh` before release.
