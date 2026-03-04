from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import sys

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from config.settings import CONFIG
from models.model_server.clients import CLIENTS, create_client


def _resolve_model_name(args: argparse.Namespace) -> str:
    if args.model:
        return args.model
    model_name = CONFIG.benchmark("level2.model", "")
    if model_name:
        return model_name
    raise SystemExit("Missing model name. Use --model or set benchmark.level2.model in config.yaml.")


def _load_level2_pipeline():
    try:
        from models.pipeline import Level2Pipeline, default_level2_config
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(
            "Level2 pipeline not found. Please implement models/pipeline/level2_pipeline.py "
            "and export Level2Pipeline/default_level2_config in models/pipeline/__init__.py."
        ) from exc
    return Level2Pipeline, default_level2_config


def main() -> None:
    parser = argparse.ArgumentParser(description="SocialOmni Benchmark Runner (Level2)")
    parser.add_argument("--model", choices=sorted(CLIENTS.keys()))
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--resume", action="store_true", help="Resume from existing output without interactive prompt")
    args = parser.parse_args()

    model_name = _resolve_model_name(args)
    test = create_client(model_name)

    Level2Pipeline, default_level2_config = _load_level2_pipeline()
    config = default_level2_config(test.model_name)
    if args.resume and config.output_path.exists():
        config = replace(config, resume=True)
    elif config.output_path.exists():
        answer = input(f"Existing results found: {config.output_path}. Resume? (y/N) ").strip().lower()
        if answer in {"y", "yes"}:
            config = replace(config, resume=True)
    if args.max_samples is not None or args.start_index:
        config = replace(
            config,
            max_samples=args.max_samples,
            start_index=args.start_index,
        )

    pipeline = Level2Pipeline(test, config)
    try:
        pipeline.run()
    except KeyboardInterrupt:
        print("\n[STOP] User interrupted Level2 benchmark.")
        raise SystemExit(130)


if __name__ == "__main__":
    main()
