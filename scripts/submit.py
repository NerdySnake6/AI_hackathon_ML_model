"""Build the submission bundle from local project files."""

import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BUNDLE = ROOT / "bundle.zip"

EXCLUDE_DIRS = {"data", ".venv", ".git", "notebooks", "__pycache__", ".ipynb_checkpoints", "scripts"}
EXCLUDE_FILES = {
    "bundle.zip", ".env", ".env.example", ".gitignore", ".python-version",
    "README.md", "pyproject.toml", "uv.lock", "tz.md", "train.csv",
}
REQUIRED_ARTIFACTS = {
    "artifacts/ct_classifier.pkl",
    "artifacts/embeddings.pkl",
    "artifacts/franchise_dict.json",
    "artifacts/knowledge_graph.json",
    "artifacts/typequery_model.pkl",
    "artifacts/metadata.json",
}


def validate_submission_state() -> None:
    """Fail fast when required artifacts are missing or stale."""
    missing_paths = [
        str(ROOT / rel_path)
        for rel_path in sorted(REQUIRED_ARTIFACTS)
        if not (ROOT / rel_path).exists()
    ]
    if missing_paths:
        raise RuntimeError(
            "Missing required submission artifacts:\n- " + "\n- ".join(missing_paths)
        )

    sys.path.insert(0, str(ROOT))
    from ct_classifier import load_artifacts as load_ct

    load_ct(str(ROOT / "artifacts" / "ct_classifier.pkl"))

    metadata_path = ROOT / "artifacts" / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if "lemmatization_enabled" not in metadata:
        raise RuntimeError(
            "Artifacts were trained with outdated metadata. "
            "Retrain with the current train.py before submitting."
        )
    if metadata["lemmatization_enabled"]:
        raise RuntimeError(
            "Submission artifacts were trained with lemmatization enabled, "
            "but the evaluation sandbox does not provide pymorphy3. "
            "Retrain with the default settings before submitting."
        )


def build_bundle() -> Path:
    """Build the submission zip after validating the local artifact state."""
    validate_submission_state()
    if BUNDLE.exists():
        BUNDLE.unlink()
    with zipfile.ZipFile(BUNDLE, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in ROOT.rglob("*"):
            if not path.is_file():
                continue
            rel = path.relative_to(ROOT)
            if any(part in EXCLUDE_DIRS for part in rel.parts):
                continue
            if rel.name in EXCLUDE_FILES:
                continue
            zf.write(path, rel.as_posix())
    print(f"built {BUNDLE} ({BUNDLE.stat().st_size} bytes)")
    return BUNDLE


def main() -> None:
    """Validate artifacts and build the submission bundle."""
    build_bundle()


if __name__ == "__main__":
    main()
