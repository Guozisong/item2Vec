#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
    echo "Usage: $0 ITEM_ID [TOPK]" >&2
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
python -m item2vec.inference query "${downstream_dir}" "$1" --top-k "${2:-10}"
