#!/usr/bin/env python3
"""Explicit GPU entry point for the original FastWAM 5B realworld baseline."""
import hydra
from fastwam.realworld.baseline_training import run_training
from fastwam.utils.config_resolvers import register_default_resolvers

register_default_resolvers()


@hydra.main(config_path="../../configs", config_name="train", version_base="1.3")
def main(cfg):
    run_training(cfg)


if __name__ == "__main__":
    main()
