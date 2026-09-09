#!/usr/bin/env bash
set -euo pipefail

if [[ $# -gt 2 ]]; then
    echo "Usage: $0 [TOPK [BLOCK_SIZE]]" >&2
    exit 2
fi

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${repository_root}/src${PYTHONPATH:+:${PYTHONPATH}}"
downstream_dir="${repository_root}/dataset/downstream"

if [[ ! -f "${downstream_dir}/trained_item.featCLS" ]]; then
    echo "Missing trained item embedding: ${downstream_dir}/trained_item.featCLS" >&2
    exit 1
fi

if [[ ! -f "${downstream_dir}/index2item.json" ]]; then
    echo "Missing item index mapping: ${downstream_dir}/index2item.json" >&2
    exit 1
fi

cd "${repository_root}"
python -m item2vec.inference export "${downstream_dir}" \
    --top-k "${1:-10}" \
    --block-size "${2:-512}"
