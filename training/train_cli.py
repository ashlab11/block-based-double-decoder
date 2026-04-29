#!/usr/bin/env python3
"""CLI entry point for single training runs with arbitrary params/tokens.

Designed to be launched via torchrun:
    torchrun --nproc_per_node=1 training/train_cli.py --params 1000000 --tokens 40000000
"""

import argparse
import os
import sys

# Ensure project root is on the path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from configs import build_config_from_dict
from configs.scaling import build_scaling_config
from training.pretrain import pretrain


def main():
    parser = argparse.ArgumentParser(
        description="Train a model with a given param count and token budget"
    )
    parser.add_argument("--params", type=int, required=True,
                        help="Target non-embedding parameter count (e.g. 1000000)")
    parser.add_argument("--tokens", type=int, required=True,
                        help="Total training tokens (e.g. 40000000)")
    args = parser.parse_args()

    cfg_dict = build_scaling_config(args.params, args.tokens)
    cfg = build_config_from_dict(cfg_dict)
    pretrain(cfg, verbose=1)


if __name__ == "__main__":
    main()
