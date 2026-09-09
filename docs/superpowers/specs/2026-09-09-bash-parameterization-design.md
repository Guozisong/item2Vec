# Bash Parameterization Design

## Goal

Expose the routine training and batch-inference settings through the existing Bash entrypoints while retaining matching Python defaults for direct module use.

## Interfaces

`scripts/train.sh` accepts zero or four positional arguments:

```bash
bash scripts/train.sh [BERT_WEIGHT WINDOW NEGATIVE EPOCHS]
```

Defaults are `0.7`, `20`, `15`, and `10`, respectively. Supplying one to three arguments is a usage error; this avoids assigning a value to the wrong training setting.

`scripts/export_similarities.sh` accepts zero to two positional arguments:

```bash
bash scripts/export_similarities.sh [TOPK BLOCK_SIZE]
```

Its defaults are `10` and `512`. `scripts/query_similar.sh ITEM_ID [TOPK]` is unchanged because a single source item does not need batch sizing.

## Implementation

Add argparse options to `item2vec.training` for BERT initialization weight, Word2Vec window, negative samples, and training epochs. Pass them into the current training function without changing any defaults or other Word2Vec settings.

Add `--block-size` to the inference export CLI and forward it to the existing blockwise exporter. The Bash scripts define the defaults and forward every selected value. Python retains the same defaults to support direct `python -m` calls.

## Validation and Documentation

Argparse enforces numeric types. Existing inference validation continues to reject invalid top-k and block sizes. Tests verify Bash argument forwarding, omitted-argument defaults, invalid training argument counts, and CLI propagation. README documents commands and parameter order.

No data paths, artifact formats, credentials, model files, or similarity output schemas change.
