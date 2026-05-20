import argparse


def parse():
    parser = argparse.ArgumentParser()

    # data preprocessing 
    parser.add_argument("--dataset_name", type=str, default="l2arctic")
    # parser.add_argument("--dataset_path", type=Path, default=DATASET_REGISTRY["l2arctic"]["path"])

    # AID 
    parser.add_argument("--source_data_path", type=str, default="data/metadata/metadaa_checked.json")
    parser.add_argument("--training_data_path", type=str, default="data/metadata/aid_train.json")
    parser.add_argument("--test_data_path", type=str, default="data/metadata/aid_test.json")

    parser.add_argument("--target_sample_rate", type=int, default=16000)

    args = parser.parse_args()


    return args