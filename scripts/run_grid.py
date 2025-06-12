#!/usr/bin/env python
import argparse
import yaml

from scripts.pipeline_grid import run_pipeline


def main():
    parser = argparse.ArgumentParser(description="Run experiment from YAML config")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Run pipeline for a single dataset config
    run_pipeline(config)


if __name__ == "__main__":
    main()
