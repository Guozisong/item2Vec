#!/usr/bin/env bash
set -euo pipefail

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${repository_root}/src${PYTHONPATH:+:${PYTHONPATH}}"
item_csv="${repository_root}/dataset/raw/item.csv"
model_dir="${repository_root}/dataset/m3e-base"
output_dir="${repository_root}/dataset/downstream"

if [[ ! -f "${item_csv}" ]]; then
    echo "Missing item CSV: ${item_csv}" >&2
    exit 1
fi

if [[ ! -d "${model_dir}" ]]; then
    echo "Missing model directory: ${model_dir}" >&2
    exit 1
fi

mkdir -p "${output_dir}"
cd "${repository_root}"
python -m item2vec.embedding "${item_csv}" "${output_dir}" "${model_dir}"
