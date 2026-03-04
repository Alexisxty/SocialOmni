#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.model_server.clients import CLIENTS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run smoke tests for all models in batch")
    parser.add_argument("--max-samples", type=int, default=1, help="Number of test samples per model, default 1")
    parser.add_argument("--server-timeout", type=int, default=900, help="Timeout in seconds for local server readiness")
    parser.add_argument("--test-timeout", type=int, default=1200, help="Timeout in seconds for each single-model test")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    models = sorted(CLIENTS.keys())

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = ROOT / "results" / "smoke_tests" / f"all_models_{run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = out_dir / "summary.csv"

    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["model", "status", "exit_code", "log_file"],
        )
        writer.writeheader()

        for model in models:
            cmd = [
                "uv",
                "run",
                "scripts/run_test_with_autoserver.py",
                "--model",
                model,
                "--max-samples",
                str(args.max_samples),
                "--server-timeout",
                str(args.server_timeout),
                "--test-timeout",
                str(args.test_timeout),
            ]
            proc = subprocess.run(  # noqa: S603
                cmd,
                cwd=str(ROOT),
                text=True,
                capture_output=True,
                check=False,
            )

            model_log = out_dir / f"{model}.log"
            model_log.write_text((proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else ""), encoding="utf-8")

            if proc.returncode == 0:
                status = "PASS"
            elif proc.returncode == 3:
                status = "FAIL_INFER"
            elif proc.returncode == 2:
                status = "FAIL_EMPTY"
            else:
                status = "FAIL"

            print(f"[{status}] {model} (exit={proc.returncode})")
            writer.writerow(
                {
                    "model": model,
                    "status": status,
                    "exit_code": proc.returncode,
                    "log_file": str(model_log),
                }
            )

    print(f"\nSummary: {summary_csv}")


if __name__ == "__main__":
    main()
