#!/usr/bin/env bash
set -euo pipefail

numeric_weight_pattern='^[+-]?([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][+-]?[0-9]+)?$'
if [[ $# -lt 1 || $# -gt 4 || ( $# -eq 4 && ${3:-} =~ ${numeric_weight_pattern} ) ]]; then
    echo "Usage: $0 ITEM_ID [TOPK [RECALL_MODE [TEXT_WEIGHT]]]" >&2
    exit 2
fi

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${repository_root}/src${PYTHONPATH:+:${PYTHONPATH}}"
downstream_dir="${repository_root}/dataset/downstream"

for artifact in item.feat1CLS behavior_item.npz; do
    if [[ ! -f "${downstream_dir}/${artifact}" ]]; then
        echo "Missing artifact: ${downstream_dir}/${artifact}; 请先生成文本向量并运行 bash scripts/train.sh" >&2
        exit 1
    fi
done

if [[ ! -f "${downstream_dir}/index2item.json" ]]; then
    echo "Missing item index mapping: ${downstream_dir}/index2item.json" >&2
    exit 1
fi

recall_mode="${3:-${RECALL_MODE:-hybrid}}"
text_weight="${4:-${TEXT_WEIGHT:-}}"
if [[ ${3:-} =~ ${numeric_weight_pattern} ]]; then
    recall_mode="${RECALL_MODE:-hybrid}"
    text_weight="$3"
fi

args=(query "${downstream_dir}" "$1"
    --top-k "${2:-10}"
    --recall-mode "${recall_mode}"
    --full-confidence-orders "${FULL_CONFIDENCE_ORDERS:-50}")
if [[ -n "${text_weight}" ]]; then
    args+=(--text-weight "${text_weight}")
fi

cd "${repository_root}"
python -m item2vec.inference "${args[@]}"
