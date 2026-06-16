from pathlib import Path


def test_project_has_required_top_level_files():
    root = Path(__file__).resolve().parents[1]
    assert (root / "pyproject.toml").exists()
    assert (root / "docker-compose.yml").exists()
    assert (root / "README.md").exists()
    assert (root / ".github" / "workflows" / "ci.yml").exists()
