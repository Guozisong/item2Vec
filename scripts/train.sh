#!/usr/bin/env bash
set -euo pipefail

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${repository_root}/src${PYTHONPATH:+:${PYTHONPATH}}"
work_dir="${repository_root}/outputs"
work_dir_set=
processed_dir=
processed_dir_set=
positionals=()
training_args=()
training_options_seen='|'

usage() {
    echo "Usage: $0 [VECTOR_SIZE MAX_BASKET_SIZE NEGATIVE EPOCHS] [--trainer attention] [--attention-heads N] [--temperature N] [--pair-loss-weight N] [--context-batch-size N] [--learning-rate N] [--validation-orders CSV] [--seed N] [--device auto|cpu|cuda] [--processed-dir DIR] [--work-dir DIR]" >&2
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --processed-dir)
            [[ $# -ge 2 && -n "$2" && "$2" != --* && -z "$processed_dir_set" ]] || { usage; exit 2; }
            processed_dir="$2"
            processed_dir_set=1
            shift 2
            ;;
        --trainer|--attention-heads|--temperature|--pair-loss-weight|--context-batch-size|--learning-rate|--validation-orders|--patience|--min-delta|--seed|--device)
            [[ $# -ge 2 && -n "$2" && "$2" != --* && "$training_options_seen" != *"|$1|"* ]] || { usage; exit 2; }
            training_options_seen="${training_options_seen}$1|"
            training_args+=("$1" "$2")
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

processed_dir="${processed_dir:-${work_dir}/processed}"
check_args=("${processed_dir}" "${work_dir}/embeddings" --check-options
    --vector-size "${vector_size}" --max-basket-size "${max_basket_size}"
    --negative "${negative}" --epochs "${epochs}" --min-order-count "${MIN_ORDER_COUNT:-5}")
if [[ ${#training_args[@]} -gt 0 ]]; then
    check_args+=("${training_args[@]}")
fi
python -m item2vec.training "${check_args[@]}"
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
python_args=("${processed_dir}" "${embeddings_dir}" --vector-size "${vector_size}"
    --max-basket-size "${max_basket_size}" --negative "${negative}" --epochs "${epochs}")
if [[ ${#training_args[@]} -gt 0 ]]; then
    python_args+=("${training_args[@]}")
fi
python_args+=(--min-order-count "${MIN_ORDER_COUNT:-5}")
python -m item2vec.training "${python_args[@]}"
