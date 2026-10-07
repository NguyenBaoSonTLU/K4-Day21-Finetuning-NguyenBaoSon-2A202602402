"""Run lab stages on Windows or Linux, retaining each stage's complete console log."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
STAGES = {f"nb{i}": p for i, p in enumerate(sorted((ROOT / "notebooks").glob("0*.py")), 1)}
STAGES["verify"] = ROOT / "scripts" / "verify.py"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stages", nargs="*", choices=list(STAGES), default=list(STAGES)[:5])
    args = parser.parse_args()
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
    logs = ROOT / "results" / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    for stage in args.stages:
        print(f"Running {stage}", flush=True)
        with (logs / f"{stage}.log").open("w", encoding="utf-8") as log:
            proc = subprocess.Popen([sys.executable, "-u", str(STAGES[stage])], cwd=ROOT,
                                    env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    text=True, encoding="utf-8", errors="replace")
            for line in proc.stdout:
                log.write(line)
                log.flush()
                print(line, end="", flush=True)
            code = proc.wait()
        if code:
            print(f"{stage} failed ({code}); see results/logs/{stage}.log", flush=True)
            return code
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
