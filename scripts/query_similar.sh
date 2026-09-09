#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 3 ]]; then
    echo "Usage: $0 ITEM_ID [TOPK [TEXT_WEIGHT]]" >&2
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

cd "${repository_root}"
python -m item2vec.inference query "${downstream_dir}" "$1" \
    --top-k "${2:-10}" \
    --text-weight "${3:-0.7}"
