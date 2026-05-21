from aid_dataset import AIDDataset, aid_collate_fn
from aid_evaluate import evaluate
from torch.utils.data import DataLoader
from parameters import parse
from aid_models import Wav2Vec2ForAID
from tqdm import tqdm
from pathlib import Path

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import logging
import datetime
import os
import json


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def set_logger(args):
    logger = logging.getLogger("current_logger")
    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime('%Y-%m-%d-%H-%M-%S')
    log_dir = args.log_dir
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_filename = os.path.join(log_dir, f"{formatted_time}.log")
    fh = logging.FileHandler(log_filename)
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger


def build_optimizer(model, args):
    model_to_optim = model.module if hasattr(model, "module") else model

    wav2vec2_params = []
    head_params = []

    for name, param in model_to_optim.named_parameters():
        if not param.requires_grad:
            continue

        if name.startswith("wav2vec2."):
            wav2vec2_params.append(param)
        else:
            head_params.append(param)

    optimizer = optim.AdamW(
        [
            {
                "params": wav2vec2_params,
                "lr": args.wav2vec2_learning_rate,
            },
            {
                "params": head_params,
                "lr": args.head_learning_rate,
            },
        ],
        weight_decay=args.weight_decay,
    )

    return optimizer

def train(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_cuda_device
    set_seed(args.seed)
    logging.basicConfig(level=logging.INFO)
    
    logger = set_logger(args)
    if torch.cuda.is_available():
        n_gpu = torch.cuda.device_count()
        logger.info(f"Number of GPUs available: {n_gpu}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    aid_model = Wav2Vec2ForAID(pretrained_model_name=args.pretrained_model_name, num_labels=args.num_labels, hidden_proj_dim=args.hidden_proj_dim, label_smoothing=args.label_smoothing)

    # for name, param in aid_model.named_parameters():
    #     logger.info(f"Parameter: {name}, requires_grad: {param.requires_grad}")

    aid_model.to(device)

    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        aid_model = nn.DataParallel(aid_model)
        logger.info(f"Using DataParallel on {torch.cuda.device_count()} GPUs")

    train_dataset = AIDDataset(args.training_data_path, target_sample_rate=args.target_sample_rate)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=aid_collate_fn,
    )

    test_dataset = AIDDataset(args.test_data_path, target_sample_rate=args.target_sample_rate)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=aid_collate_fn,
    )

    if args.small_sample:
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            collate_fn=aid_collate_fn,
            sampler=torch.utils.data.SubsetRandomSampler(range(100)),
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=aid_collate_fn,
            sampler=torch.utils.data.SubsetRandomSampler(range(100)),
        )

    optimizer = build_optimizer(aid_model, args)
    # NOTE: debug code for failure to converge: check number of trainable parameters and optimizer parameters
    # num_trainable = sum(p.numel() for p in aid_model.parameters() if p.requires_grad)
    # num_optim = sum(p.numel() for group in optimizer.param_groups for p in group["params"])

    # logger.info(f"Trainable parameters: {num_trainable}")
    # logger.info(f"Optimizer parameters: {num_optim}")
    # NOTE: end of debug code for failure to converge
    
    max_epochs = args.max_epochs
    
    # NOTE: debug code for failure to converge: check which parameters require gradients
    # for name, param in aid_model.named_parameters():
    #     if any(key in name for key in ["feature_extractor", "encoder.layers", "classifier", "proj"]):
    #         logger.info(f"{name}: requires_grad={param.requires_grad}")
    # NOTE: end of debug code for failure to converge
    
    # NOTE: when intending to block the training loop, comment out the triple quotes below 
    # """
    for epoch in range(max_epochs):
        tr_loss = 0.0
        num_train_examples, num_train_steps = 0, 0
        logger.info(f"Training Epoch {epoch+1}/{max_epochs}")
        
        for step, batch in enumerate(tqdm(train_loader)):
            num_train_steps += 1
            num_train_examples += batch["input_values"].size(0)
            aid_model.train()
            optimizer.zero_grad()
            input_values = batch["input_values"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = aid_model(input_values=input_values, attention_mask=attention_mask, labels=labels)
            loss = outputs["loss"]
            if loss.dim() > 0:
                loss = loss.mean()
            tr_loss += loss.item()
            loss.backward()
            # NOTE: debug code for failure to converge: log gradient norms of classifier, projection, and layer norm parameters
            # for name, param in aid_model.named_parameters():
            #     if param.requires_grad and param.grad is not None:
            #         if "classifier" in name or "proj" in name or "layer_norm" in name:
            #             logger.info(f"{name}: grad_norm={param.grad.norm().item():.6f}")
            # NOTE: end of debug code for failure to converge
            optimizer.step()
        logger.info(f"Epoch {epoch+1} - Average Training Loss: {tr_loss / len(train_loader):.4f}")
        # """
        # NOTE: end of training loop
    
    # NOTE: debug code for failure to converge: one-batch overfit test
    # aid_model.train()
    # batch = next(iter(train_loader))
    # input_values = batch["input_values"].to(device)
    # attention_mask = batch["attention_mask"].to(device)
    # labels = batch["labels"].to(device)
    # for step in range(100):
    #     optimizer.zero_grad()
    #     outputs = aid_model(
    #         input_values=input_values,
    #         attention_mask=attention_mask,
    #         labels=labels,
    #     )
    #     loss = outputs["loss"]
    #     if loss.dim() > 0:
    #         loss = loss.mean()
    #     loss.backward()
    #     optimizer.step()
    #     if step % 10 == 0:
    #         logits = outputs["logits"]
    #         preds = logits.argmax(dim=-1)
    #         acc = (preds == labels).float().mean().item()
    #         logger.info(
    #             f"step={step}, loss={loss.item():.4f}, acc={acc:.4f}"
    #         )
    # NOTE: end of one-batch overfit test

    label2id_path = Path("data/metadata/label2id.json")

    with label2id_path.open("r", encoding="utf-8") as f:
        label2id = json.load(f)

    id2label = {v: k for k, v in label2id.items()}
    
    test_result = evaluate(
        model=aid_model,
        data_loader=test_loader,
        device=device,
        id2label=id2label,
    )

    logger.info(
        f"Epoch {epoch + 1} - "
        f"Test Loss: {test_result['loss']:.4f}, "
        f"Acc: {test_result['accuracy']:.4f}, "
        f"Macro P/R/F1: "
        f"{test_result['macro_precision']:.4f} / "
        f"{test_result['macro_recall']:.4f} / "
        f"{test_result['macro_f1']:.4f}, "
        f"Weighted P/R/F1: "
        f"{test_result['weighted_precision']:.4f} / "
        f"{test_result['weighted_recall']:.4f} / "
        f"{test_result['weighted_f1']:.4f}"
    )

    for label_name, metrics in test_result["per_class_metrics"].items():
        logger.info(
            f"{label_name}: "
            f"P={metrics['precision']:.4f}, "
            f"R={metrics['recall']:.4f}, "
            f"F1={metrics['f1']:.4f}, "
            f"support={metrics['support']}"
        )

    logger.info("\n" + test_result["classification_report"])
    logger.info(f"\nConfusion Matrix:\n{test_result['confusion_matrix']}")

    logger.info(f"Saving model checkpoint to {model_dir}")
    checkpoint_path = model_dir / f"aid_model_epoch_{epoch+1}.pt"

    model_to_save = aid_model.module if hasattr(aid_model, "module") else aid_model

    torch.save(
        {
            "epoch": epoch + 1,
            "model_state_dict": model_to_save.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            # "train_loss": avg_train_loss,
            # "test_acc": test_acc,
        },
        checkpoint_path,
    )
    logger.info(f"Model checkpoint saved to {checkpoint_path}")
    
    



    # sanity check
    # batch = next(iter(train_loader))
    # print(batch["input_values"].shape)
    # print(batch["attention_mask"].shape)
    # print(batch["labels"].shape)
    # print(batch["labels"])
    # print(batch["utt_ids"][:3])
    # print(batch["speaker_ids"][:3])
    # print(batch["accent_labels"][:3])



if __name__ == "__main__":
    args = parse()
    train(args)