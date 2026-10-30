#!/usr/bin/env bash
set -euo pipefail

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${repository_root}/src${PYTHONPATH:+:${PYTHONPATH}}"
work_dir="${repository_root}/outputs"
work_dir_set=
positionals=()
candidate_bias_weight=.4
candidate_bias_weight_set=

usage() {
    echo "Usage: $0 [VECTOR_SIZE MAX_BASKET_SIZE NEGATIVE EPOCHS [CANDIDATE_BIAS_WEIGHT]] [--candidate-bias-weight N] [--work-dir DIR]" >&2
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --candidate-bias-weight)
            [[ $# -ge 2 && -n "$2" && "$2" != --* && -z "${candidate_bias_weight_set}" ]] || { usage; exit 2; }
            candidate_bias_weight="$2"
            candidate_bias_weight_set=1
            shift 2
            ;;
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
    4|5)
        vector_size="$1"
        max_basket_size="$2"
        negative="$3"
        epochs="$4"
        if [[ $# -eq 5 ]]; then
            [[ -z "${candidate_bias_weight_set}" ]] || { usage; exit 2; }
            candidate_bias_weight="$5"
        fi
        ;;
    *)
        usage
        exit 1
        ;;
esac

processed_dir="${work_dir}/processed"
embeddings_dir="${work_dir}/embeddings"
for artifact in baskets.jsonl support.npz manifest.json; do
    if [[ ! -f "${processed_dir}/${artifact}" ]]; then
        echo "Missing processed basket artifact: ${processed_dir}/${artifact}; 请先运行 bash scripts/preprocess_baskets.sh" >&2
        exit 1
    fi
done
if [[ ! -f "${embeddings_dir}/item2index.json" || ! -f "${embeddings_dir}/index2item.json" ]]; then
    echo "Missing item index mappings in ${embeddings_dir}" >&2
    exit 1
fi

mkdir -p "${embeddings_dir}"
echo "工作目录：${work_dir}"
cd "${repository_root}"
python -m item2vec.training "${processed_dir}" "${embeddings_dir}" \
    --vector-size "${vector_size}" \
    --max-basket-size "${max_basket_size}" \
    --negative "${negative}" \
    --epochs "${epochs}" \
    --candidate-bias-weight "${candidate_bias_weight}" \
    --min-order-count "${MIN_ORDER_COUNT:-5}"
