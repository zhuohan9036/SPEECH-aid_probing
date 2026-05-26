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
import copy


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
    return logger, formatted_time


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
    
    logger, log_filename = set_logger(args)
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

    train_dataset = AIDDataset(args.training_data_path, apply_perturbation=args.apply_perturbation, target_sample_rate=args.target_sample_rate, perturbation_prob=args.perturbation_prob)
    
    # NOTE: debug code for checking raw train_dataset items
    # import time

    # print("Start checking raw train_dataset items")

    # for i in range(100):
    #     t0 = time.time()
    #     print(f"Before get item {i}", flush=True)
    #     item = train_dataset[i]
    #     print(
    #         f"After get item {i}, time={time.time() - t0:.2f}s, "
    #         f"keys={item.keys()}",
    #         flush=True
    #     )
    # NOTE: end debug block
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=aid_collate_fn,
    )

    test_dataset = AIDDataset(args.test_data_path, apply_perturbation=False, target_sample_rate=args.target_sample_rate, perturbation_prob=args.perturbation_prob)
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
    
    # NOTE: sanity check for data loading and label correctness
    # batch = next(iter(test_loader))
    # print(batch["labels"][:20])
    # from json_file_operations import get_json_content
    # test_metadata = get_json_content(args.test_data_path)

    # for item in test_metadata[:20]:
    #     print(
    #         item["speaker_id"],
    #         item["accent_label"],
    #         item["label_id"],
    #         item["utt_id"],
    #     )
    # NOTE: end of sanity check for data loading and label correctness

    optimizer = build_optimizer(aid_model, args)
    # NOTE: debug code for failure to converge: check number of trainable parameters and optimizer parameters
    # num_trainable = sum(p.numel() for p in aid_model.parameters() if p.requires_grad)
    # num_optim = sum(p.numel() for group in optimizer.param_groups for p in group["params"])

    # logger.info(f"Trainable parameters: {num_trainable}")
    # logger.info(f"Optimizer parameters: {num_optim}")
    # NOTE: end of debug code for failure to converge
    
    max_epochs = args.max_epochs

    # NOTE: comment out the following block to check id2label
    # #%%
    # from pathlib import Path
    # import json
    
    label2id_path = Path("data/metadata/label2id.json")

    with label2id_path.open("r", encoding="utf-8") as f:
        label2id = json.load(f)

    id2label = {v: k for k, v in label2id.items()}
    # print(id2label)
    # #%%
    # NOTE: end of code to check id2label

    # NOTE: code for saving the best model
    best_macro_f1 = -1.0
    best_epoch = -1
    best_model_state_dict = None
    best_optimizer_state_dict = None
    best_train_loss = None
    best_test_result = None
    # NOTE: end of code for saving the best model
    
    
    # NOTE: debug code for failure to converge: check which parameters require gradients
    # for name, param in aid_model.named_parameters():
    #     if any(key in name for key in ["feature_extractor", "encoder.layers", "classifier", "proj"]):
    #         logger.info(f"{name}: requires_grad={param.requires_grad}")
    # NOTE: end of debug code for failure to converge
    
    # NOTE: when intending to block the training loop, comment out the triple quotes below 
    # """
    for epoch in range(max_epochs):
    # for epoch in range(1):  # NOTE: debug code for failure to converge: run only one epoch
        tr_loss = 0.0
        num_train_examples, num_train_steps = 0, 0
        logger.info(f"Training Epoch {epoch+1}/{max_epochs}")
        
        for step, batch in enumerate(tqdm(train_loader)):
            logger.info(f"Got batch {step}")
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
        
        # NOTE: uncomment the following block to run evaluation after each epoch
        test_result = evaluate(
            model=aid_model,
            data_loader=test_loader,
            device=device,
            id2label=id2label,
        )

        current_macro_f1 = test_result["macro_f1"]

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
        
        if current_macro_f1 > best_macro_f1:
            best_macro_f1 = current_macro_f1
            best_epoch = epoch + 1
            best_train_loss = tr_loss / len(train_loader)
            best_test_result = test_result

            model_to_save = aid_model.module if hasattr(aid_model, "module") else aid_model

            best_model_state_dict = copy.deepcopy(model_to_save.state_dict())
            best_optimizer_state_dict = copy.deepcopy(optimizer.state_dict())

            logger.info(
                f"New best model found at epoch {best_epoch}. "
                f"Macro-F1={best_macro_f1:.4f}. "
                f"Checkpoint will be saved after training finishes."
            )

        else:
            logger.info(
                f"Epoch {epoch + 1} did not improve best Macro-F1. "
                f"Current Macro-F1={current_macro_f1:.4f}, "
                f"Best Macro-F1={best_macro_f1:.4f} at epoch {best_epoch}."
            )
            # NOTE: end of code for testing after each epoch
        
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

    # NOTE: uncomment the following block to run evaluation after training
    # test_result = evaluate(
    #     model=aid_model,
    #     data_loader=test_loader,
    #     device=device,
    #     id2label=id2label,
    # )

    # logger.info(
    #     f"Epoch {epoch + 1} - "
    #     f"Test Loss: {test_result['loss']:.4f}, "
    #     f"Acc: {test_result['accuracy']:.4f}, "
    #     f"Macro P/R/F1: "
    #     f"{test_result['macro_precision']:.4f} / "
    #     f"{test_result['macro_recall']:.4f} / "
    #     f"{test_result['macro_f1']:.4f}, "
    #     f"Weighted P/R/F1: "
    #     f"{test_result['weighted_precision']:.4f} / "
    #     f"{test_result['weighted_recall']:.4f} / "
    #     f"{test_result['weighted_f1']:.4f}"
    # )

    # for label_name, metrics in test_result["per_class_metrics"].items():
    #     logger.info(
    #         f"{label_name}: "
    #         f"P={metrics['precision']:.4f}, "
    #         f"R={metrics['recall']:.4f}, "
    #         f"F1={metrics['f1']:.4f}, "
    #         f"support={metrics['support']}"
    #     )

    # logger.info("\n" + test_result["classification_report"])
    # logger.info(f"\nConfusion Matrix:\n{test_result['confusion_matrix']}")

    # logger.info(f"Saving model checkpoint to {model_dir}")
    # checkpoint_path = model_dir / f"aid_model_epoch_{epoch+1}_{log_filename}.pt"

    # model_to_save = aid_model.module if hasattr(aid_model, "module") else aid_model

    # torch.save(
    #     {
    #         "epoch": epoch + 1,
    #         "model_state_dict": model_to_save.state_dict(),
    #         "optimizer_state_dict": optimizer.state_dict(),
    #         # "train_loss": avg_train_loss,
    #         # "test_acc": test_acc,
    #     },
    #     checkpoint_path,
    # )
    # logger.info(f"Model checkpoint saved to {checkpoint_path}")
    # NOTE: end of evaluation and checkpoint saving
    
    # NOTE: code for saving the best model checkpoint
    if best_model_state_dict is not None:
        checkpoint_path = (
            model_dir
            / f"aid_model_best_epoch_{best_epoch}_{log_filename}.pt"
        )

        logger.info(f"Saving best model checkpoint to {checkpoint_path}")

        torch.save(
            {
                "epoch": best_epoch,
                "model_state_dict": best_model_state_dict,
                "optimizer_state_dict": best_optimizer_state_dict,
                "train_loss": best_train_loss,
                "test_loss": best_test_result["loss"],
                "test_acc": best_test_result["accuracy"],
                "macro_precision": best_test_result["macro_precision"],
                "macro_recall": best_test_result["macro_recall"],
                "macro_f1": best_test_result["macro_f1"],
                "weighted_precision": best_test_result["weighted_precision"],
                "weighted_recall": best_test_result["weighted_recall"],
                "weighted_f1": best_test_result["weighted_f1"],
                "label2id": label2id,
                "id2label": id2label,
            },
            checkpoint_path,
        )

        logger.info(
            f"Best model checkpoint saved. "
            f"Best epoch={best_epoch}, Best Macro-F1={best_macro_f1:.4f}"
        )
    else:
        logger.warning("No best model was found. Check whether training/evaluation ran correctly.")
    # NOTE: end of code for saving the best model checkpoint


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