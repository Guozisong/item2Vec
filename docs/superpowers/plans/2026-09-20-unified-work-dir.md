# Unified Work Directory Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let every Bash stage read and write pipeline artifacts below one configurable `--work-dir`, organized as `raw/`, `embeddings/`, and `results/`.

**Architecture:** Bash scripts own CLI parsing and resolve the default work directory to `<repository>/outputs`. They pass explicit absolute directories into Python modules. Fetch writes raw data, embedding and training share the embeddings directory, and inference reads embeddings while writing CSV files to results.

**Tech Stack:** Bash, Python 3.13, argparse, pathlib, pytest

---

## File map

- Modify `src/item2vec/data_fetch.py`: accept an explicit raw output directory.
- Modify `src/item2vec/inference.py`: separate artifact input directory from result output directory.
- Modify `scripts/fetch_data.sh`: parse `--work-dir` and `--env-file`.
- Modify `scripts/generate_embeddings.sh`: parse `--work-dir` and `--model-dir`.
- Modify `scripts/train.sh`: parse `--work-dir` alongside existing training positionals.
- Modify `scripts/query_similar.sh`: parse `--work-dir` alongside existing query positionals.
- Modify `scripts/export_similarities.sh`: parse `--work-dir` alongside existing export positionals.
- Modify `scripts/run_pipeline.sh`: parse and forward the common paths.
- Modify `tests/test_data_fetch.py`, `tests/test_inference.py`, and `tests/test_scripts.py`: prove the new path contracts and preserve existing behavior.
- Modify `.gitignore` and `README.md`: document and ignore the new default output layout.

### Task 1: Make Python input and output directories explicit

**Files:**
- Modify: `src/item2vec/data_fetch.py`
- Modify: `src/item2vec/inference.py`
- Test: `tests/test_data_fetch.py`
- Test: `tests/test_inference.py`

- [ ] **Step 1: Write failing fetch and inference path tests**

Add a fetch test that calls `data_fetch.main([str(raw_dir)])`, stubs `fetch_data`, and asserts that the explicit directory is created and forwarded. Add inference tests that call `query_item(artifact_dir, ..., output_dir=result_dir)` and `export_all(artifact_dir, ..., output_dir=result_dir)`, then assert returned files are below `result_dir`.

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```bash
python3.13 -m pytest -q tests/test_data_fetch.py tests/test_inference.py
```

Expected: failures because `data_fetch.main` takes no arguments and inference writes into the artifact directory.

- [ ] **Step 3: Implement the explicit Python path contracts**

Change fetch to:

```python
def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir")
    args = parser.parse_args(argv)
    os.makedirs(args.output_dir, exist_ok=True)
    fetch_data(args.output_dir, os.environ["ALI_ACCESS_ID"], os.environ["ALI_SECRET_ACCESS_KEY"])
```

Add `output_dir=None` to `query_item` and `export_all`; default it to the artifact directory for direct Python API compatibility, create it, and write result CSV files there. Add optional `--output-dir` to both inference subcommands and forward it from `main`.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run the same pytest command. Expected: all focused tests pass.

### Task 2: Add work directory support to individual Bash stages

**Files:**
- Modify: `scripts/fetch_data.sh`
- Modify: `scripts/generate_embeddings.sh`
- Modify: `scripts/train.sh`
- Modify: `scripts/query_similar.sh`
- Modify: `scripts/export_similarities.sh`
- Test: `tests/test_scripts.py`

- [ ] **Step 1: Update test helpers to create the new default layout**

Change stub artifacts from `dataset/raw` and `dataset/downstream` to `outputs/raw` and `outputs/embeddings`. Extend inference stubs to accept `--output-dir`, and capture the forwarded arguments.

- [ ] **Step 2: Write failing custom work directory tests**

For each stage, invoke the copied script with `--work-dir <tmp_path/custom>`. Assert:

```text
fetch:      Python receives <custom>/raw
embedding:  Python receives <custom>/raw/item.csv and <custom>/embeddings
train:      Python receives <custom>/raw and <custom>/embeddings
query:      Python receives <custom>/embeddings and --output-dir <custom>/results
export:     Python receives <custom>/embeddings and --output-dir <custom>/results
```

Add tests proving `--env-file` and `--model-dir` override only their external inputs. Add parametrized cases for option placement before and after positionals and for unknown, duplicate, or missing-value options.

- [ ] **Step 3: Run script tests and verify RED**

Run:

```bash
python3.13 -m pytest -q tests/test_scripts.py
```

Expected: failures because scripts reject the new options or continue using `dataset/downstream`.

- [ ] **Step 4: Implement minimal Bash parsing**

Each script resolves:

```bash
work_dir="${repository_root}/outputs"
positionals=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --work-dir)
            [[ $# -ge 2 && -n "$2" ]] || { usage; exit 2; }
            [[ -z "${work_dir_set:-}" ]] || { usage; exit 2; }
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
set -- "${positionals[@]}"
```

`fetch_data.sh` additionally parses `--env-file`; `generate_embeddings.sh` parses `--model-dir`. All scripts print `工作目录：<path>`, use `raw/`, `embeddings/`, and `results/`, and create only directories they write.

- [ ] **Step 5: Run script tests and verify GREEN**

Run the focused script tests. Expected: all pass.

### Task 3: Forward paths through the complete pipeline

**Files:**
- Modify: `scripts/run_pipeline.sh`
- Test: `tests/test_scripts.py`

- [ ] **Step 1: Write failing pipeline forwarding tests**

Change stub stages to log all received arguments. Run:

```text
bash run_pipeline.sh --work-dir /pai/output --env-file /pai/config/.env --model-dir /pai/model
```

Assert exact calls:

```text
fetch --work-dir /pai/output --env-file /pai/config/.env
generate --work-dir /pai/output --model-dir /pai/model
train --work-dir /pai/output
```

Also assert unknown, duplicate, and missing-value options stop before any stage runs.

- [ ] **Step 2: Run pipeline tests and verify RED**

Run the relevant `tests/test_scripts.py` tests. Expected: the existing pipeline ignores or misroutes options.

- [ ] **Step 3: Implement pipeline option parsing and forwarding**

Parse only `--work-dir`, `--env-file`, and `--model-dir`. Build separate argument arrays:

```bash
fetch_args=(--work-dir "$work_dir")
embedding_args=(--work-dir "$work_dir")
train_args=(--work-dir "$work_dir")
[[ -n "$env_file" ]] && fetch_args+=(--env-file "$env_file")
[[ -n "$model_dir" ]] && embedding_args+=(--model-dir "$model_dir")
```

Invoke each stage in order and preserve `set -euo pipefail` behavior.

- [ ] **Step 4: Run pipeline and script tests and verify GREEN**

Run:

```bash
python3.13 -m pytest -q tests/test_scripts.py
```

Expected: all script tests pass.

### Task 4: Update defaults and user documentation

**Files:**
- Modify: `.gitignore`
- Modify: `README.md`
- Test: `tests/test_scripts.py`

- [ ] **Step 1: Write the failing ignore rule assertion**

Change the existing Git ignore test to require `outputs/` while continuing to require the local model ignore rule.

- [ ] **Step 2: Run the test and verify RED**

Run:

```bash
python3.13 -m pytest -q tests/test_scripts.py::test_gitignore_excludes_local_model_assets
```

Expected: failure because `.gitignore` does not contain `outputs/`.

- [ ] **Step 3: Update ignore rules and README**

Add `outputs/` to `.gitignore`. Update README directory diagrams, full pipeline examples, individual command syntax, PAI mount examples, defaults, external input options, and result locations. Remove references to generated artifacts below `dataset/downstream`.

- [ ] **Step 4: Run all verification**

Run:

```bash
python3.13 -m pytest -q
git diff --check
```

Expected: all tests pass and `git diff --check` returns no output.

- [ ] **Step 5: Review the final diff**

Confirm only the planned scripts, Python path interfaces, tests, ignore rules, README, spec, and plan changed. Preserve existing untracked files.
