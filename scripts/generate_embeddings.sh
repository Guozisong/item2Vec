#!/usr/bin/env bash
set -euo pipefail

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${repository_root}/src${PYTHONPATH:+:${PYTHONPATH}}"
work_dir="${repository_root}/outputs"
model_dir="${repository_root}/dataset/m3e-base"
work_dir_set=
model_dir_set=

usage() {
    echo "Usage: $0 [--work-dir DIR] [--model-dir DIR]" >&2
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --work-dir)
            [[ $# -ge 2 && -n "$2" && "$2" != --* && -z "${work_dir_set}" ]] || { usage; exit 2; }
            work_dir="$2"
            work_dir_set=1
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

item_csv="${work_dir}/raw/item.csv"
embeddings_dir="${work_dir}/embeddings"
if [[ ! -f "${item_csv}" ]]; then
    echo "Missing item CSV: ${item_csv}" >&2
    exit 1
fi
if [[ ! -d "${model_dir}" ]]; then
    echo "Missing model directory: ${model_dir}" >&2
    exit 1
fi

mkdir -p "${embeddings_dir}"
echo "工作目录：${work_dir}"
cd "${repository_root}"
python -m item2vec.embedding "${item_csv}" "${embeddings_dir}" "${model_dir}"
