#!/usr/bin/env bash
set -euo pipefail

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${repository_root}/src${PYTHONPATH:+:${PYTHONPATH}}"
raw_data_dir="${repository_root}/dataset/raw"
downstream_dir="${repository_root}/dataset/downstream"

if [[ ! -f "${raw_data_dir}/order_item.csv" ]]; then
    echo "Missing order item CSV: ${raw_data_dir}/order_item.csv" >&2
    exit 1
fi

if [[ ! -f "${downstream_dir}/item.feat1CLS" ]]; then
    echo "Missing item embedding: ${downstream_dir}/item.feat1CLS" >&2
    exit 1
fi

if [[ ! -f "${downstream_dir}/item2index.json" || ! -f "${downstream_dir}/index2item.json" ]]; then
    echo "Missing item index mappings" >&2
    exit 1
fi

cd "${repository_root}"
python -m item2vec.training "${raw_data_dir}" "${downstream_dir}"
