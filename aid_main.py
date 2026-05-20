from dataset import AIDDataset, aid_collate_fn
from torch.utils.data import DataLoader
from parameters import parse


def train(args):
    print("is it here?")
    train_dataset = AIDDataset(args.training_data_path, target_sample_rate=args.target_sample_rate)
    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        collate_fn=aid_collate_fn,
    )

    test_dataset = AIDDataset(args.test_data_path, target_sample_rate=args.target_sample_rate)
    test_loader = DataLoader(
        test_dataset,
        batch_size=8,
        shuffle=False,
        collate_fn=aid_collate_fn,
    )

    # sanity check
    batch = next(iter(train_loader))
    print(batch["input_values"].shape)
    print(batch["attention_mask"].shape)
    print(batch["labels"].shape)
    print(batch["labels"])
    print(batch["utt_ids"][:3])
    print(batch["speaker_ids"][:3])
    print(batch["accent_labels"][:3])

if __name__ == "__main__":
    args = parse()
    train(args)