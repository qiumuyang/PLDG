import argparse
from pathlib import Path


def parse_args(extra: dict[str, type] = {}) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("-c",
                        "--config",
                        help="Path to configuration file",
                        type=Path,
                        nargs="+",
                        required=True)
    parser.add_argument("-s",
                        "--seed",
                        help="Random seed",
                        type=int,
                        default=0)
    parser.add_argument("-d",
                        "--domain",
                        help="Target domain id",
                        type=int,
                        required=True)
    parser.add_argument("-o",
                        "--output",
                        help="Directory to save logs and checkpoints",
                        type=Path,
                        required=True)
    for k, v in extra.items():
        if v is bool:
            parser.add_argument(f"--{k}", action="store_true")
        else:
            parser.add_argument(f"--{k}", type=v)
    return parser.parse_args()


def parse_eval_args(extra: dict[str, type] = {}) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path, help="Path to the checkpoint")
    parser.add_argument("-c",
                        "--config",
                        help="Path to configuration file",
                        type=Path,
                        nargs="+",
                        required=True)
    parser.add_argument("-s",
                        "--seed",
                        help="Random seed",
                        type=int,
                        default=0)
    parser.add_argument("-d",
                        "--domain",
                        help="Target domain id",
                        type=lambda s: [int(x) for x in s.split(",")],
                        required=True)
    parser.add_argument("--split",
                        help="Split to evaluate",
                        default="val",
                        choices=["train", "val"])
    parser.add_argument("-o", "--output", type=Path, default=None)
    for k, v in extra.items():
        if v is bool:
            parser.add_argument(f"--{k}", action="store_true")
        else:
            parser.add_argument(f"--{k}", type=v)
    return parser.parse_args()
