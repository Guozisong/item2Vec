#!/usr/bin/env bash
set -euo pipefail

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${repository_root}/src${PYTHONPATH:+:${PYTHONPATH}}"
work_dir="${repository_root}/outputs"
env_file=
model_dir=
work_dir_set=
env_file_set=
model_dir_set=
source_args=()
min_basket_args=()
candidate_bias_args=()
training_args=()
training_options_seen='|'

usage() {
    echo "Usage: $0 --item-input-{csv|table|query-file} SOURCE --order-input-{csv|table|query-file} SOURCE [--min-basket-size N] [--candidate-bias-weight N] [--work-dir DIR] [--env-file FILE] [--model-dir DIR]" >&2
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --trainer|--basket-pair-budget|--negative-exclusion|--hard-negative-ratio|--hard-negative-warmup|--hard-negative-pool-size|--pair-batch-size|--validation-orders|--patience|--min-delta|--seed|--device)
            [[ $# -ge 2 && -n "$2" && "$2" != --* && "$training_options_seen" != *"|$1|"* ]] || { usage; exit 2; }
            training_options_seen="${training_options_seen}$1|"
            training_args+=("$1" "$2")
            shift 2
            ;;
        --candidate-bias-weight)
            [[ $# -ge 2 && -n "$2" && "$2" != --* && ${#candidate_bias_args[@]} -eq 0 ]] || { usage; exit 2; }
            candidate_bias_args=(--candidate-bias-weight "$2")
            shift 2
            ;;
        --min-basket-size)
            [[ $# -ge 2 && -n "$2" && "$2" != --* && ${#min_basket_args[@]} -eq 0 ]] || { usage; exit 2; }
            min_basket_args=(--min-basket-size "$2")
            shift 2
            ;;
        --item-input-csv|--item-input-table|--item-input-query-file|--order-input-csv|--order-input-table|--order-input-query-file)
            [[ $# -ge 2 && -n "$2" && "$2" != --* ]] || { usage; exit 2; }
            source_args+=("$1" "$2")
            shift 2
            ;;
        --work-dir)
            [[ $# -ge 2 && -n "$2" && "$2" != --* && -z "${work_dir_set}" ]] || { usage; exit 2; }
            work_dir="$2"
            work_dir_set=1
            shift 2
            ;;
        --env-file)
            [[ $# -ge 2 && -n "$2" && "$2" != --* && -z "${env_file_set}" ]] || { usage; exit 2; }
            env_file="$2"
            env_file_set=1
            shift 2
            ;;
        --model-dir)
            [[ $# -ge 2 && -n "$2" && "$2" != --* && -z "${model_dir_set}" ]] || { usage; exit 2; }
            model_dir="$2"
            model_dir_set=1
            shift 2
            ;;
        *)
            usage
            exit 2
            ;;
    esac
done

fetch_args=(--work-dir "${work_dir}")
if [[ ${#source_args[@]} -gt 0 ]]; then
    fetch_args+=("${source_args[@]}")
fi
embedding_args=(--work-dir "${work_dir}")
train_args=(--work-dir "${work_dir}")
if [[ ${#candidate_bias_args[@]} -gt 0 ]]; then
    train_args+=("${candidate_bias_args[@]}")
fi
if [[ ${#training_args[@]} -gt 0 ]]; then
    train_args+=("${training_args[@]}")
fi
preprocess_args=(--work-dir "${work_dir}")
if [[ ${#min_basket_args[@]} -gt 0 ]]; then
    preprocess_args+=("${min_basket_args[@]}")
fi
if [[ -n "${env_file}" ]]; then
    fetch_args+=(--env-file "${env_file}")
fi
if [[ -n "${model_dir}" ]]; then
    embedding_args+=(--model-dir "${model_dir}")
fi

# Check training options before any fetch, embedding or preprocessing writes.
check_args=("${work_dir}/processed" "${work_dir}/embeddings" --check-options)
if [[ ${#candidate_bias_args[@]} -gt 0 ]]; then
    check_args+=("${candidate_bias_args[@]}")
fi
if [[ ${#training_args[@]} -gt 0 ]]; then
    check_args+=("${training_args[@]}")
fi
python -m item2vec.training "${check_args[@]}"

bash "${repository_root}/scripts/fetch_data.sh" "${fetch_args[@]}"
bash "${repository_root}/scripts/generate_embeddings.sh" "${embedding_args[@]}"
bash "${repository_root}/scripts/preprocess_baskets.sh" "${preprocess_args[@]}"
bash "${repository_root}/scripts/train.sh" "${train_args[@]}"
