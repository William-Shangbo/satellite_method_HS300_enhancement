from pathlib import Path

from index_platform.common.settings import load_settings
from index_platform.ingest.contracts import load_dataset_contracts


def test_dataset_contracts_include_index_weight():
    project_root = Path(__file__).resolve().parents[1]
    contracts = load_dataset_contracts(project_root / "configs" / "datasets" / "tushare_pipeline.yaml")
    assert "index_weight" in {item.api_name for item in contracts.datasets}
    assert "daily" in {item.api_name for item in contracts.datasets}


def test_runtime_settings_default_to_project_relative_paths():
    project_root = Path(__file__).resolve().parents[1]
    settings = load_settings(project_root)
    assert settings.project_root == project_root
    assert settings.raw_data_root == project_root / "data" / "raw"
    assert settings.artifact_root == project_root / "artifacts"
