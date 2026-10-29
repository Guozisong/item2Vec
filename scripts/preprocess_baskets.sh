#!/usr/bin/env bash
set -euo pipefail

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${repository_root}/src${PYTHONPATH:+:${PYTHONPATH}}"
work_dir="${repository_root}/outputs"
work_dir_set=
positionals=()

usage() {
    echo "Usage: $0 [MAX_BASKET_SIZE [MIN_CO_BASKET_ORDERS]] [--work-dir DIR]" >&2
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --work-dir)
            [[ $# -ge 2 && -n "$2" && "$2" != --* && -z "${work_dir_set}" ]] || { usage; exit 2; }
            work_dir="$2"
            work_dir_set=1
            shift 2
            ;;
        --*) usage; exit 2 ;;
        *) positionals+=("$1"); shift ;;
    esac
done

if [[ ${#positionals[@]} -gt 2 ]]; then
    usage
    exit 2
fi
max_basket_size="${positionals[0]:-30}"
min_co_basket_orders="${positionals[1]:-${MIN_ORDER_COUNT:-5}}"
raw_dir="${work_dir}/raw"
embeddings_dir="${work_dir}/embeddings"
processed_dir="${work_dir}/processed"
for path in "${raw_dir}/order_item.csv" "${embeddings_dir}/item2index.json" "${embeddings_dir}/index2item.json"; do
    if [[ ! -f "${path}" ]]; then
        echo "Missing preprocessing input: ${path}" >&2
        exit 1
    fi
done

echo "工作目录：${work_dir}"
cd "${repository_root}"
python -m item2vec.preprocessing "${raw_dir}" "${embeddings_dir}" "${processed_dir}" \
    --max-basket-size "${max_basket_size}" \
    --min-co-basket-orders "${min_co_basket_orders}"
