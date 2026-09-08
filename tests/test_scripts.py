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
