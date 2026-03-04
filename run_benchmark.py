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
from models.pipeline import Level1Pipeline, default_level1_config
from models.model_server.clients import CLIENTS, create_client


def _resolve_model_name(args: argparse.Namespace) -> str:
    if args.model:
        return args.model
    model_name = CONFIG.benchmark("level1.model", "")
    if model_name:
        return model_name
    raise SystemExit("Missing model name. Use --model or set benchmark.level1.model in config.yaml.")


def main() -> None:
    parser = argparse.ArgumentParser(description="SocialOmni Benchmark Runner")
    parser.add_argument("--model", choices=sorted(CLIENTS.keys()))
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=0)
    args = parser.parse_args()

    model_name = _resolve_model_name(args)
    test = create_client(model_name)

    config = default_level1_config(test.model_name)
    if config.output_path.exists():
        answer = input(f"Existing results found: {config.output_path}. Resume? (y/N) ").strip().lower()
        if answer in {"y", "yes"}:
            config = replace(config, resume=True)
    if args.max_samples is not None or args.start_index:
        config = replace(
            config,
            max_samples=args.max_samples,
            start_index=args.start_index,
        )

    pipeline = Level1Pipeline(test, config)
    pipeline.run()


if __name__ == "__main__":
    main()
