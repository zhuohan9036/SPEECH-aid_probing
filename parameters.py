import argparse
from pathlib import Path

DATASET_REGISTRY = {
    "l2arctic": {
        "path": Path("~/speech/l2arctic"),
    },
    "cmu_arctic": {
        "path": Path("~/speech/cmu_arctic"),
    },
}


def parse():
    parser = argparse.ArgumentParser()

    # data preprocessing 
    parser.add_argument("--dataset_name", type=str, default="l2arctic")
    parser.add_argument("--dataset_path", type=Path, default=DATASET_REGISTRY["l2arctic"]["path"])

    args = parser.parse_args()


    return args