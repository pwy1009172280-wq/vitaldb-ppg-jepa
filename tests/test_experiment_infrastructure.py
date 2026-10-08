import json

from src.config import PipelineConfig, load_config
from src.experiments import ExperimentRegistry, create_run
from src.experiments.metadata import save_config


def test_config_save_and_load_round_trip(tmp_path):
    config = PipelineConfig(
        dataset={"pretrain": {"name": "test-fixture-ppg", "modality": "PPG"}},
        preprocessing={"pretrain": {"transforms": []}},
        experiment={"seed": 13, "smoke_only": True},
    )
    path = tmp_path / "config.yaml"
    save_config(config, path)
    loaded = load_config(path)
    assert loaded.dataset == config.dataset
    assert loaded.preprocessing == config.preprocessing
    assert loaded.experiment == config.experiment


def test_experiment_run_metadata_and_layout(tmp_path):
    run, manifest = create_run(
        "synthetic_smoke",
        {"dataset": {"name": "synthetic"}, "experiment": {"seed": 7}},
        "dataset-manifest://synthetic/v1",
        "split://synthetic/split-7",
        7,
        results_root=tmp_path / "results",
        repo_root=tmp_path,
    )
    assert run.config_path.exists()
    assert run.manifest_path.exists()
    assert run.metrics_path.exists()
    assert run.checkpoints_path.is_dir()
    assert run.logs_path.is_dir()
    saved = json.loads(run.manifest_path.read_text(encoding="utf-8"))
    assert saved["experiment_name"] == "synthetic_smoke"
    assert saved["dataset_manifest_ref"] == "dataset-manifest://synthetic/v1"
    assert saved["subject_split_ref"] == "split://synthetic/split-7"
    assert saved["seed"] == 7
    assert saved["timestamp_utc"]
    assert manifest.to_dict() == saved


def test_experiment_registry_is_explicit_and_duplicate_safe():
    registry = ExperimentRegistry()
    registry.register("synthetic", lambda: "definition")
    assert registry.names() == ("synthetic",)
    assert registry.get("synthetic")() == "definition"
    try:
        registry.register("synthetic", lambda: "other")
    except ValueError as error:
        assert "already" in str(error)
    else:
        raise AssertionError("duplicate experiment registration should fail")
