#!/usr/bin/env bash
set -euo pipefail

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
work_dir="${repository_root}/outputs"
env_file=
model_dir=
work_dir_set=
env_file_set=
model_dir_set=

usage() {
    echo "Usage: $0 [--work-dir DIR] [--env-file FILE] [--model-dir DIR]" >&2
}

while [[ $# -gt 0 ]]; do
    case "$1" in
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
embedding_args=(--work-dir "${work_dir}")
train_args=(--work-dir "${work_dir}")
if [[ -n "${env_file}" ]]; then
    fetch_args+=(--env-file "${env_file}")
fi
if [[ -n "${model_dir}" ]]; then
    embedding_args+=(--model-dir "${model_dir}")
fi

bash "${repository_root}/scripts/fetch_data.sh" "${fetch_args[@]}"
bash "${repository_root}/scripts/generate_embeddings.sh" "${embedding_args[@]}"
bash "${repository_root}/scripts/train.sh" "${train_args[@]}"
