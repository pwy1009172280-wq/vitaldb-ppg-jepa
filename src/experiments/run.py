"""Run directory creation with no training or data-loading responsibilities."""

from pathlib import Path
import subprocess

from .manifest import ExperimentRunManifest
from .metadata import collect_environment_info, save_config, save_json_metadata, seed_everything


def detect_git_commit(repo_root: str | Path | None = None) -> str | None:
    """Return the current git commit hash when git is available."""
    cwd = Path(repo_root) if repo_root else Path.cwd()
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=cwd, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def detect_code_version(repo_root: str | Path | None = None) -> str | None:
    """Return the current git commit, with ``-dirty`` when applicable."""
    commit = detect_git_commit(repo_root)
    state = detect_git_state(repo_root)
    if commit is None:
        return None
    return f"{commit}-dirty" if state == "dirty" else commit


def detect_git_state(repo_root: str | Path | None = None) -> str | None:
    """Return a compact clean/dirty git state when git is available."""
    cwd = Path(repo_root) if repo_root else Path.cwd()
    try:
        dirty = subprocess.call(
            ["git", "diff", "--quiet"], cwd=cwd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        ) != 0
        return "dirty" if dirty else "clean"
    except (OSError, subprocess.SubprocessError):
        return None


class RunDirectory:
    """Paths and metadata for the canonical experiment output layout."""

    def __init__(self, path: Path) -> None:
        self.path = path

    @property
    def config_path(self) -> Path:
        return self.path / "config.yaml"

    @property
    def manifest_path(self) -> Path:
        return self.path / "manifest.json"

    @property
    def metrics_path(self) -> Path:
        return self.path / "metrics.json"

    @property
    def checkpoints_path(self) -> Path:
        return self.path / "checkpoints"

    @property
    def logs_path(self) -> Path:
        return self.path / "logs"


def create_run(
    experiment_name: str,
    resolved_config: object,
    dataset_manifest_ref: str,
    subject_split_ref: str,
    seed: int,
    *,
    results_root: str | Path = "results",
    repo_root: str | Path | None = None,
    overwrite: bool = False,
    preprocessing_version: str | None = None,
    evaluation_protocol=None,
    pretraining_subject_ids: tuple[str, ...] = (),
) -> tuple[RunDirectory, ExperimentRunManifest]:
    """Create and initialize a run directory and its reproducibility metadata."""
    if not experiment_name or Path(experiment_name).name != experiment_name:
        raise ValueError("experiment_name must be a single safe directory name")
    run = RunDirectory(Path(results_root) / experiment_name)
    if run.path.exists() and any(run.path.iterdir()) and not overwrite:
        raise FileExistsError(f"run directory already contains files: {run.path}")
    run.path.mkdir(parents=True, exist_ok=True)
    run.checkpoints_path.mkdir(exist_ok=True)
    run.logs_path.mkdir(exist_ok=True)
    manifest = ExperimentRunManifest(
        experiment_name=experiment_name,
        resolved_config=resolved_config,
        dataset_manifest_ref=dataset_manifest_ref,
        subject_split_ref=subject_split_ref,
        seed=seed,
        code_version=detect_code_version(repo_root),
        preprocessing_version=preprocessing_version,
        environment=collect_environment_info(),
        git_commit=detect_git_commit(repo_root),
        git_state=detect_git_state(repo_root),
        evaluation_protocol=evaluation_protocol,
        pretraining_subject_ids=pretraining_subject_ids,
    )
    save_config(resolved_config, run.config_path)
    save_json_metadata(manifest, run.manifest_path)
    save_json_metadata({}, run.metrics_path)
    seed_everything(seed)
    return run, manifest
