import json
import os
import sys
import time
import zipfile
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.environ.get("API_KEY")
APP_BASE_URL = os.environ.get("APP_BASE_URL", "https://app.ai-business-spb.ru").rstrip("/")
CASE = "mediascope"
ROOT = Path(__file__).resolve().parent.parent
BUNDLE = ROOT / "bundle.zip"

EXCLUDE_DIRS = {"data", ".venv", ".git", "notebooks", "__pycache__", ".ipynb_checkpoints", "scripts"}
EXCLUDE_FILES = {
    "bundle.zip", ".env", ".env.example", ".gitignore", ".python-version",
    "README.md", "pyproject.toml", "uv.lock", "QWEN.md", "tz.md", "train.csv",
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


def submit(bundle: Path) -> int:
    """Upload the prepared bundle to the submission API."""
    if not API_KEY:
        print("ERROR: API_KEY not set", file=sys.stderr)
        sys.exit(1)
    url = f"{APP_BASE_URL}/api/{CASE}/submissions"
    with open(bundle, "rb") as f:
        r = requests.post(
            url,
            headers={"X-API-Key": API_KEY},
            files={"file": (bundle.name, f, "application/zip")},
            timeout=300,
        )
    r.raise_for_status()
    sub = r.json()
    sub_id = sub.get("id")
    print(f"submitted: id={sub_id} status={sub.get('status')}")
    return int(sub_id)


def poll(sub_id: int) -> None:
    """Poll the submission API until the run finishes."""
    url = f"{APP_BASE_URL}/api/{CASE}/submissions/{sub_id}"
    while True:
        r = requests.get(url, headers={"X-API-Key": API_KEY}, timeout=60)
        r.raise_for_status()
        sub = r.json()
        status = sub.get("status")
        print(f"  status={status}")
        if status in ("completed", "failed", "timeout", "error"):
            print(f"score: {sub.get('score')}")
            print(f"details: {sub.get('score_details')}")
            if sub.get("error_log"):
                print(f"error_log: {sub.get('error_log')}")
            return
        time.sleep(10)


def main() -> None:
    """Validate artifacts, build the bundle, submit it, and poll for results."""
    bundle = build_bundle()
    sub_id = submit(bundle)
    poll(sub_id)


if __name__ == "__main__":
    main()
