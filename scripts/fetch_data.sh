#!/usr/bin/env bash
set -euo pipefail

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${repository_root}/src${PYTHONPATH:+:${PYTHONPATH}}"
work_dir="${repository_root}/outputs"
credentials_file="${repository_root}/dataset/raw/.env"
work_dir_set=
env_file_set=
item_source_set=
order_source_set=
needs_odps=
source_args=()

usage() {
    echo "Usage: $0 --item-input-{csv|table|query-file} SOURCE --order-input-{csv|table|query-file} SOURCE [--work-dir DIR] [--env-file FILE]" >&2
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --item-input-csv|--item-input-table|--item-input-query-file)
            [[ $# -ge 2 && -n "$2" && "$2" != --* && -z "${item_source_set}" ]] || { usage; exit 2; }
            item_source_set=1
            [[ "$1" == --item-input-csv ]] || needs_odps=1
            source_value="$2"
            if [[ "$1" != --item-input-table && "$source_value" != /* ]]; then
                source_value="${PWD}/${source_value}"
            fi
            source_args+=("$1" "$source_value")
            shift 2
            ;;
        --order-input-csv|--order-input-table|--order-input-query-file)
            [[ $# -ge 2 && -n "$2" && "$2" != --* && -z "${order_source_set}" ]] || { usage; exit 2; }
            order_source_set=1
            [[ "$1" == --order-input-csv ]] || needs_odps=1
            source_value="$2"
            if [[ "$1" != --order-input-table && "$source_value" != /* ]]; then
                source_value="${PWD}/${source_value}"
            fi
            source_args+=("$1" "$source_value")
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
            credentials_file="$2"
            env_file_set=1
            shift 2
            ;;
        *)
            usage
            exit 2
            ;;
    esac
done

[[ -n "${item_source_set}" && -n "${order_source_set}" ]] || { usage; exit 2; }
if [[ -n "${needs_odps}" ]]; then
    if [[ ! -f "${credentials_file}" ]]; then
        echo "Missing credentials file: ${credentials_file}" >&2
        exit 1
    fi
    set -a
    source "${credentials_file}"
    set +a
    if [[ -z "${ALI_ACCESS_ID:-${access_id:-}}" || -z "${ALI_SECRET_ACCESS_KEY:-${access_key:-}}" || -z "${ALI_PROJECT:-${project:-}}" ]]; then
        echo "Missing ODPS credentials" >&2
        exit 1
    fi
fi

raw_dir="${work_dir}/raw"
mkdir -p "${raw_dir}"
echo "工作目录：${work_dir}"
cd "${repository_root}"
python -m item2vec.data_fetch "${raw_dir}" "${source_args[@]}"
