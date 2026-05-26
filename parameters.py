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

    parser.add_argument("--pretrained_model_name", type=str, default="facebook/wav2vec2-base")

    parser.add_argument("--target_sample_rate", type=int, default=16000)
    parser.add_argument("--num_labels", type=int, default=7)
    parser.add_argument("--hidden_proj_dim", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=48)

    parser.add_argument("--apply_perturbation", action="store_true")
    parser.add_argument("--perturbation_prob", type=float, default=0.75)


    # GPU
    parser.add_argument("--visible_cuda_device", type=str, default="0,1")

    # training 
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_dir", type=str, default="logs")
    parser.add_argument("--model_dir", type=str, default="pts")
    parser.add_argument("--num_workers", type=int, default=0)
    # parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--wav2vec2_learning_rate", type=float, default=1e-5)
    parser.add_argument("--head_learning_rate", type=float, default=3e-3)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_epochs", type=int, default=50)
    parser.add_argument("--label_smoothing", type=float, default=0.25)
    parser.add_argument("--small_sample", action="store_true")

    args = parser.parse_args()


    return args