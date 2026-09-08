#!/usr/bin/env bash
set -euo pipefail

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

bash "${repository_root}/scripts/fetch_data.sh"
bash "${repository_root}/scripts/generate_embeddings.sh"
bash "${repository_root}/scripts/train.sh"
