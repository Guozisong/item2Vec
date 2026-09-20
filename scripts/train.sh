#!/usr/bin/env bash
set -euo pipefail

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${repository_root}/src${PYTHONPATH:+:${PYTHONPATH}}"
work_dir="${repository_root}/outputs"
work_dir_set=
positionals=()

usage() {
    echo "Usage: $0 [VECTOR_SIZE MAX_BASKET_SIZE NEGATIVE EPOCHS] [--work-dir DIR]" >&2
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --work-dir)
            [[ $# -ge 2 && -n "$2" && "$2" != --* && -z "${work_dir_set}" ]] || { usage; exit 2; }
            work_dir="$2"
            work_dir_set=1
            shift 2
            ;;
        --*)
            usage
            exit 2
            ;;
        *)
            positionals+=("$1")
            shift
            ;;
    esac
done
if [[ ${#positionals[@]} -gt 0 ]]; then
    set -- "${positionals[@]}"
else
    set --
fi

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

raw_data_dir="${work_dir}/raw"
embeddings_dir="${work_dir}/embeddings"
if [[ ! -f "${raw_data_dir}/order_item.csv" ]]; then
    echo "Missing order item CSV: ${raw_data_dir}/order_item.csv" >&2
    exit 1
fi
if [[ ! -f "${embeddings_dir}/item2index.json" || ! -f "${embeddings_dir}/index2item.json" ]]; then
    echo "Missing item index mappings in ${embeddings_dir}" >&2
    exit 1
fi

mkdir -p "${embeddings_dir}"
echo "工作目录：${work_dir}"
cd "${repository_root}"
python -m item2vec.training "${raw_data_dir}" "${embeddings_dir}" \
    --vector-size "${vector_size}" \
    --max-basket-size "${max_basket_size}" \
    --negative "${negative}" \
    --epochs "${epochs}" \
    --min-order-count "${MIN_ORDER_COUNT:-5}"
