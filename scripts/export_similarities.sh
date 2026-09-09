#!/usr/bin/env bash
set -euo pipefail

if [[ $# -gt 3 ]]; then
    echo "Usage: $0 [TOPK [BLOCK_SIZE [TEXT_WEIGHT]]]" >&2
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
python -m item2vec.inference export "${downstream_dir}" \
    --top-k "${1:-10}" \
    --block-size "${2:-512}" \
    --text-weight "${3:-0.7}"
