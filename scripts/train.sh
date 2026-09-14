#!/usr/bin/env bash
set -euo pipefail

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${repository_root}/src${PYTHONPATH:+:${PYTHONPATH}}"
raw_data_dir="${repository_root}/dataset/raw"
downstream_dir="${repository_root}/dataset/downstream"

usage() {
    echo "Usage: $0 [VECTOR_SIZE MAX_BASKET_SIZE NEGATIVE EPOCHS]" >&2
}

case "$#" in
    0)
        vector_size=128
        max_basket_size=30
        negative=15
        epochs=10
        ;;
    4)
        vector_size="$1"
        max_basket_size="$2"
        negative="$3"
        epochs="$4"
        ;;
    *)
        usage
        exit 1
        ;;
esac

if [[ ! -f "${raw_data_dir}/order_item.csv" ]]; then
    echo "Missing order item CSV: ${raw_data_dir}/order_item.csv" >&2
    exit 1
fi

if [[ ! -f "${downstream_dir}/item2index.json" || ! -f "${downstream_dir}/index2item.json" ]]; then
    echo "Missing item index mappings" >&2
    exit 1
fi

cd "${repository_root}"
python -m item2vec.training "${raw_data_dir}" "${downstream_dir}" \
    --vector-size "${vector_size}" \
    --max-basket-size "${max_basket_size}" \
    --negative "${negative}" \
    --epochs "${epochs}" \
    --min-order-count "${MIN_ORDER_COUNT:-5}"
