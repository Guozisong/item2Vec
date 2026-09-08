import subprocess
from pathlib import Path


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
