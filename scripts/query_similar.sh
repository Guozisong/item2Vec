#!/usr/bin/env bash
set -euo pipefail

numeric_weight_pattern='^[+-]?([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][+-]?[0-9]+)?$'
repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${repository_root}/src${PYTHONPATH:+:${PYTHONPATH}}"
work_dir="${repository_root}/outputs"
work_dir_set=
positionals=()

usage() {
    echo "Usage: $0 ITEM_ID [TOPK [RECALL_MODE [TEXT_WEIGHT]]] [--work-dir DIR]" >&2
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

if [[ $# -lt 1 || $# -gt 4 || ( $# -eq 4 && ${3:-} =~ ${numeric_weight_pattern} ) ]]; then
    usage
    exit 2
fi

embeddings_dir="${work_dir}/embeddings"
results_dir="${work_dir}/results"
for artifact in item.feat1CLS behavior_item.npz; do
    if [[ ! -f "${embeddings_dir}/${artifact}" ]]; then
        echo "Missing artifact: ${embeddings_dir}/${artifact}; 请先生成文本向量并运行 bash scripts/train.sh" >&2
        exit 1
    fi
done
if [[ ! -f "${embeddings_dir}/index2item.json" ]]; then
    echo "Missing item index mapping: ${embeddings_dir}/index2item.json" >&2
    exit 1
fi

recall_mode="${3:-${RECALL_MODE:-hybrid}}"
text_weight="${4:-${TEXT_WEIGHT:-}}"
if [[ ${3:-} =~ ${numeric_weight_pattern} ]]; then
    recall_mode="${RECALL_MODE:-hybrid}"
    text_weight="$3"
fi

mkdir -p "${results_dir}"
echo "工作目录：${work_dir}"
args=(query "${embeddings_dir}" "$1"
    --output-dir "${results_dir}"
    --top-k "${2:-10}"
    --recall-mode "${recall_mode}"
    --full-confidence-orders "${FULL_CONFIDENCE_ORDERS:-50}")
if [[ -n "${text_weight}" ]]; then
    args+=(--text-weight "${text_weight}")
fi

cd "${repository_root}"
python -m item2vec.inference "${args[@]}"
