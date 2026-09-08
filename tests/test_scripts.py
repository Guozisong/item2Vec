import subprocess
import sys
import types
from pathlib import Path

import pytest

from item2vec import data_fetch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


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
