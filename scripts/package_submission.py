"""Create an Option A ZIP from an explicit allowlist (no secrets or model cache)."""
from pathlib import Path
import argparse
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student-id", default="2A202602402")
    args = parser.parse_args()
    if not args.student_id.isalnum():
        parser.error("student ID must be alphanumeric")
    name = "lab21_" + args.student_id
    destination = ROOT / "submission" / (name + ".zip")
    selected = []
    patterns = {
        "submission": ("*.md", "*.txt"),
        "results": ("*.json", "*.csv", "logs/*.log"),
        "adapters/correct": ("*.json", "*.safetensors", "*.jinja", "*.txt", "*.md"),
        "notebooks": ("*.py",),
        "colab": ("*.ipynb",),
        "docs": ("*.md",),
        "src/labkit": ("*.py",),
        "scripts": ("*.py",),
        "tests": ("*.py",),
    }
    for directory, globs in patterns.items():
        for pattern in globs:
            selected.extend(p for p in (ROOT / directory).glob(pattern) if p.is_file())
    for rel in ("requirements.txt", "requirements-cpu.txt", "pyproject.toml", "README.md", "rubric.md",
                "HARDWARE-GUIDE.md", "BONUS-CHALLENGE.md", "BONUS-CHALLENGE-EN.md", "LICENSE",
                "Makefile", ".env.example",
                "data/train_seed.jsonl", "data/eval_target.jsonl", "data/eval_regression.jsonl", "data/checksums.json"):
        selected.append(ROOT / rel)
    for required in ("submission/REPORT.md", "results/verdict.json", "results/runs.csv",
                     "adapters/correct/adapter_model.safetensors", "adapters/correct/adapter_config.json"):
        if not (ROOT / required).is_file():
            raise SystemExit(f"Missing {required}; complete and verify the experiment first")
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(set(selected)):
            archive.write(path, str(Path(name) / path.relative_to(ROOT)))
    with zipfile.ZipFile(destination) as archive:
        assert archive.testzip() is None
        members = [Path(member) for member in archive.namelist()]
        assert not any(member.name == ".env" or member.name == "holdout_secret.jsonl"
                       or ".cache" in member.parts or ".venv" in member.parts
                       for member in members), "Submission archive contains a secret or cache file"
    print(f"Wrote {destination} ({destination.stat().st_size / 1024**2:.1f} MiB)")


if __name__ == "__main__":
    main()
