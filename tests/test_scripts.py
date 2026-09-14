import os
import subprocess
import sys
import types
from pathlib import Path

import pytest

from item2vec import data_fetch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPOSITORY_ROOT / "scripts"


def _write_stage(path, name, exit_code=0):
    path.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        f"printf '%s\\n' '{name}' >> \"${{PIPELINE_LOG}}\"\n"
        f"exit {exit_code}\n"
    )


def _run_pipeline_with_stubs(tmp_path, generate_exit_code=0):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    pipeline = scripts / "run_pipeline.sh"
    pipeline.write_text((SCRIPTS / "run_pipeline.sh").read_text())
    _write_stage(scripts / "fetch_data.sh", "fetch")
    _write_stage(scripts / "generate_embeddings.sh", "generate", generate_exit_code)
    _write_stage(scripts / "train.sh", "train")
    log = tmp_path / "pipeline.log"
    result = subprocess.run(
        ["/bin/bash", str(pipeline)],
        text=True,
        capture_output=True,
        env={**os.environ, "PIPELINE_LOG": str(log)},
    )
    return result, log.read_text().splitlines()


def test_run_pipeline_executes_stages_in_order(tmp_path):
    result, stages = _run_pipeline_with_stubs(tmp_path)

    assert result.returncode == 0
    assert stages == ["fetch", "generate", "train"]


def test_run_pipeline_stops_after_failed_middle_stage(tmp_path):
    result, stages = _run_pipeline_with_stubs(tmp_path, generate_exit_code=17)

    assert result.returncode == 17
    assert stages == ["fetch", "generate"]


def test_gitignore_excludes_local_model_assets():
    ignored_paths = (REPOSITORY_ROOT / ".gitignore").read_text().splitlines()
    assert "dataset/m3e-base/" in ignored_paths


def test_generate_embeddings_rejects_missing_item_csv(tmp_path):
    result = subprocess.run(
        [str(REPOSITORY_ROOT / "scripts" / "generate_embeddings.sh")],
        cwd=tmp_path,
        text=True,
        capture_output=True,
    )

    assert result.returncode != 0
    assert "Missing item CSV" in result.stderr


def test_fetch_data_uses_default_endpoint_when_environment_value_is_blank(monkeypatch, tmp_path):
    captured = {}

    class FakeODPS:
        def __init__(self, access_id, access_key, project, endpoint):
            captured['endpoint'] = endpoint

        def execute_sql(self, sql):
            raise AssertionError('ODPS query must not run in this test')

    monkeypatch.setitem(sys.modules, 'odps', types.SimpleNamespace(ODPS=FakeODPS))
    monkeypatch.setenv('ALI_PROJECT', 'test-project')
    monkeypatch.setenv('ALI_ENDPOINT', '')

    with pytest.raises(AssertionError, match='ODPS query'):
        data_fetch.fetch_data(tmp_path, 'id', 'key')

    assert captured['endpoint'] == data_fetch.DEFAULT_ENDPOINT


def test_data_fetch_main_rejects_missing_project_before_fetch(monkeypatch):
    monkeypatch.setenv('ALI_ACCESS_ID', 'id')
    monkeypatch.setenv('ALI_SECRET_ACCESS_KEY', 'key')
    monkeypatch.delenv('ALI_PROJECT', raising=False)
    monkeypatch.setattr(data_fetch, 'fetch_data', lambda *args: pytest.fail('fetch must not run'))

    with pytest.raises(RuntimeError, match='Missing ODPS credentials'):
        data_fetch.main()


def _run_inference_script_with_stub(tmp_path, script_name, arguments, environment=None):
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    script = scripts_dir / script_name
    script.write_text((SCRIPTS / script_name).read_text())
    script.chmod(0o755)

    package_dir = tmp_path / "src" / "item2vec"
    package_dir.mkdir(parents=True)
    (package_dir / "__init__.py").write_text("")
    (package_dir / "inference.py").write_text(
        "import os\n"
        "import sys\n"
        "from pathlib import Path\n"
        "Path(os.environ['INFERENCE_ARGS_LOG']).write_text('\\n'.join(sys.argv[1:]))\n"
    )
    downstream_dir = tmp_path / "dataset" / "downstream"
    downstream_dir.mkdir(parents=True)
    (downstream_dir / "item.feat1CLS").write_bytes(b"vectors")
    (downstream_dir / "behavior_item.npz").write_bytes(b"vectors")
    (downstream_dir / "index2item.json").write_text("{}")
    log = tmp_path / "inference-args.log"

    result = subprocess.run(
        [str(script), *arguments],
        text=True,
        capture_output=True,
        env={
            **{key: value for key, value in os.environ.items()
               if key not in {"RECALL_MODE", "FULL_CONFIDENCE_ORDERS", "TEXT_WEIGHT"}},
            "INFERENCE_ARGS_LOG": str(log),
            **(environment or {}),
        },
    )
    return result, log.read_text().splitlines() if log.exists() else []


@pytest.mark.parametrize(
    ("arguments", "expected_top_k"),
    [(["A/../B"], "10"), (["A/../B", "7"], "7")],
)
def test_query_script_forwards_item_id_and_top_k(tmp_path, arguments, expected_top_k):
    result, arguments = _run_inference_script_with_stub(
        tmp_path, "query_similar.sh", arguments
    )

    assert result.returncode == 0
    assert arguments == [
        "query",
        str(tmp_path / "dataset" / "downstream"),
        "A/../B",
        "--top-k",
        expected_top_k,
        "--recall-mode",
        "hybrid",
        "--full-confidence-orders",
        "50",
    ]


@pytest.mark.parametrize(
    ("arguments", "expected_top_k", "expected_block_size"),
    [([], "10", "512"), (["6"], "6", "512"), (["6", "128"], "6", "128")],
)
def test_export_script_forwards_top_k_and_block_size(
    tmp_path, arguments, expected_top_k, expected_block_size
):
    result, arguments = _run_inference_script_with_stub(
        tmp_path, "export_similarities.sh", arguments
    )

    assert result.returncode == 0
    assert arguments == [
        "export",
        str(tmp_path / "dataset" / "downstream"),
        "--top-k",
        expected_top_k,
        "--block-size",
        expected_block_size,
        "--recall-mode",
        "hybrid",
        "--full-confidence-orders",
        "50",
    ]


@pytest.mark.parametrize(
    ("script_name", "base_arguments", "expected_prefix"),
    [
        ("query_similar.sh", ["A", "7"], ["A", "--top-k", "7"]),
        ("export_similarities.sh", ["6", "128"],
         ["--top-k", "6", "--block-size", "128"]),
    ],
)
@pytest.mark.parametrize(
    ("extra_arguments", "environment", "expected_options"),
    [
        ([], {"RECALL_MODE": "complement", "FULL_CONFIDENCE_ORDERS": "80",
              "TEXT_WEIGHT": "0.2"},
         ["--recall-mode", "complement", "--full-confidence-orders", "80",
          "--text-weight", "0.2"]),
        (["similar"], {"RECALL_MODE": "complement", "TEXT_WEIGHT": "0.3"},
         ["--recall-mode", "similar", "--full-confidence-orders", "50",
          "--text-weight", "0.3"]),
        (["similar", "0.4"], {"RECALL_MODE": "complement", "TEXT_WEIGHT": "0.2"},
         ["--recall-mode", "similar", "--full-confidence-orders", "50",
          "--text-weight", "0.4"]),
        ([], {"RECALL_MODE": "", "FULL_CONFIDENCE_ORDERS": "", "TEXT_WEIGHT": ""},
         ["--recall-mode", "hybrid", "--full-confidence-orders", "50"]),
        (["invalid-mode"], {},
         ["--recall-mode", "invalid-mode", "--full-confidence-orders", "50"]),
    ],
)
def test_inference_scripts_forward_recall_overrides(
    tmp_path, script_name, base_arguments, expected_prefix,
    extra_arguments, environment, expected_options
):
    result, forwarded = _run_inference_script_with_stub(
        tmp_path, script_name, base_arguments + extra_arguments, environment
    )

    assert result.returncode == 0
    assert forwarded == [
        "query" if script_name == "query_similar.sh" else "export",
        str(tmp_path / "dataset" / "downstream"),
        *expected_prefix,
        *expected_options,
    ]


@pytest.mark.parametrize(
    ("script_name", "arguments"),
    [("query_similar.sh", []),
     ("query_similar.sh", ["A", "2", "hybrid", "0.7", "extra"]),
     ("export_similarities.sh", ["2", "512", "hybrid", "0.7", "too-many"])],
)
def test_inference_scripts_reject_invalid_argument_counts(tmp_path, script_name, arguments):
    result, forwarded = _run_inference_script_with_stub(tmp_path, script_name, arguments)

    assert result.returncode != 0
    assert "Usage:" in result.stderr
    assert forwarded == []


def _run_train_script_with_stub(tmp_path, arguments, environment=None):
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    script = scripts_dir / "train.sh"
    script.write_text((SCRIPTS / "train.sh").read_text())
    script.chmod(0o755)

    package_dir = tmp_path / "src" / "item2vec"
    package_dir.mkdir(parents=True)
    (package_dir / "__init__.py").write_text("")
    (package_dir / "training.py").write_text(
        "import os\n"
        "import sys\n"
        "from pathlib import Path\n"
        "Path(os.environ['TRAINING_ARGS_LOG']).write_text('\\n'.join(sys.argv[1:]))\n"
    )
    raw_dir = tmp_path / "dataset" / "raw"
    raw_dir.mkdir(parents=True)
    (raw_dir / "order_item.csv").write_text("user_id,dt,prod_id\n")
    downstream_dir = tmp_path / "dataset" / "downstream"
    downstream_dir.mkdir(parents=True)
    (downstream_dir / "item.feat1CLS").write_bytes(b"vectors")
    (downstream_dir / "item2index.json").write_text("{}")
    (downstream_dir / "index2item.json").write_text("{}")
    log = tmp_path / "training-args.log"

    result = subprocess.run(
        [str(script), *arguments],
        text=True,
        capture_output=True,
        env={
            **{key: value for key, value in os.environ.items() if key != "MIN_ORDER_COUNT"},
            "TRAINING_ARGS_LOG": str(log),
            **(environment or {}),
        },
    )
    return result, log.read_text().splitlines() if log.exists() else []


def test_train_script_forwards_default_parameters(tmp_path):
    result, arguments = _run_train_script_with_stub(tmp_path, [])

    assert result.returncode == 0
    assert arguments == [
        str(tmp_path / "dataset" / "raw"), str(tmp_path / "dataset" / "downstream"),
        "--vector-size", "128", "--max-basket-size", "30", "--negative", "15",
        "--epochs", "10", "--min-order-count", "5",
    ]


def test_train_script_forwards_custom_parameters(tmp_path):
    result, arguments = _run_train_script_with_stub(tmp_path, ["64", "30", "4", "6"])

    assert result.returncode == 0
    assert arguments == [
        str(tmp_path / "dataset" / "raw"), str(tmp_path / "dataset" / "downstream"),
        "--vector-size", "64", "--max-basket-size", "30", "--negative", "4",
        "--epochs", "6", "--min-order-count", "5",
    ]


@pytest.mark.parametrize("arguments", [[], ["64", "30", "4", "6"]])
@pytest.mark.parametrize(("min_order_count", "expected"), [("9", "9"), ("", "5")])
def test_train_script_forwards_min_order_count_environment(
    tmp_path, arguments, min_order_count, expected
):
    result, forwarded = _run_train_script_with_stub(
        tmp_path, arguments, {"MIN_ORDER_COUNT": min_order_count}
    )

    assert result.returncode == 0
    assert forwarded[-2:] == ["--min-order-count", expected]


@pytest.mark.parametrize("arguments", [["0.7"], ["0.7", "20"], ["0.7", "20", "15"], ["0.7", "20", "15", "10", "extra"]])
def test_train_script_rejects_partial_or_extra_parameter_sets(tmp_path, arguments):
    result, forwarded = _run_train_script_with_stub(tmp_path, arguments)

    assert result.returncode != 0
    assert "Usage:" in result.stderr
    assert forwarded == []
