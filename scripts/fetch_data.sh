#!/usr/bin/env bash
set -euo pipefail

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${repository_root}/src${PYTHONPATH:+:${PYTHONPATH}}"
work_dir="${repository_root}/outputs"
credentials_file="${repository_root}/dataset/raw/.env"
work_dir_set=
env_file_set=

usage() {
    echo "Usage: $0 [--work-dir DIR] [--env-file FILE]" >&2
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

if [[ ! -f "${credentials_file}" ]]; then
    echo "Missing credentials file: ${credentials_file}" >&2
    exit 1
fi
set -a
source "${credentials_file}"
set +a
if [[ -z "${ALI_ACCESS_ID:-}" || -z "${ALI_SECRET_ACCESS_KEY:-}" || -z "${ALI_PROJECT:-}" ]]; then
    echo "Missing ODPS credentials" >&2
    exit 1
fi

raw_dir="${work_dir}/raw"
mkdir -p "${raw_dir}"
echo "工作目录：${work_dir}"
cd "${repository_root}"
python -m item2vec.data_fetch "${raw_dir}"
