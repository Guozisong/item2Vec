#!/usr/bin/env bash
set -euo pipefail

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${repository_root}/src${PYTHONPATH:+:${PYTHONPATH}}"
credentials_file="${repository_root}/dataset/raw/.env"

if [[ ! -f "${credentials_file}" ]]; then
    echo "Missing credentials file: ${credentials_file}" >&2
    exit 1
fi

set -a
source "${credentials_file}"
set +a

if [[ -z "${ALI_ACCESS_ID:-}" || -z "${ALI_SECRET_ACCESS_KEY:-}" ]]; then
    echo "Missing ODPS credentials" >&2
    exit 1
fi

cd "${repository_root}"
python -m item2vec.data_fetch
