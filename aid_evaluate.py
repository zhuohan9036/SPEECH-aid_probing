import torch
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)


@torch.no_grad()
def evaluate(model, data_loader, device, id2label=None):
    model.eval()

    total_loss = 0.0
    all_preds = []
    all_labels = []

    for batch in data_loader:
        input_values = batch["input_values"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(
            input_values=input_values,
            attention_mask=attention_mask,
            labels=labels,
        )

        loss = outputs["loss"]
        logits = outputs["logits"]
        preds = torch.argmax(logits, dim=-1)

        if loss.dim() > 0:
            loss = loss.mean()

        total_loss += loss.item()
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / len(data_loader)
    accuracy = accuracy_score(all_labels, all_preds)

    if id2label is not None:
        label_ids = list(range(len(id2label)))
        target_names = [id2label[i] for i in label_ids]
    else:
        label_ids = sorted(set(all_labels) | set(all_preds))
        target_names = None

    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        all_labels,
        all_preds,
        labels=label_ids,
        average="macro",
        zero_division=0,
    )

    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        all_labels,
        all_preds,
        labels=label_ids,
        average="weighted",
        zero_division=0,
    )

    per_class_precision, per_class_recall, per_class_f1, per_class_support = (
        precision_recall_fscore_support(
            all_labels,
            all_preds,
            labels=label_ids,
            average=None,
            zero_division=0,
        )
    )

    report = classification_report(
        all_labels,
        all_preds,
        labels=label_ids,
        target_names=target_names,
        digits=4,
        zero_division=0,
    )

    cm = confusion_matrix(
        all_labels,
        all_preds,
        labels=label_ids,
    )

    per_class_metrics = {}
    for i, label_id in enumerate(label_ids):
        label_name = id2label[label_id] if id2label is not None else str(label_id)
        per_class_metrics[label_name] = {
            "precision": float(per_class_precision[i]),
            "recall": float(per_class_recall[i]),
            "f1": float(per_class_f1[i]),
            "support": int(per_class_support[i]),
        }

    return {
        "loss": avg_loss,
        "accuracy": accuracy,

        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,

        "weighted_precision": weighted_precision,
        "weighted_recall": weighted_recall,
        "weighted_f1": weighted_f1,

        "per_class_metrics": per_class_metrics,

        "preds": all_preds,
        "labels": all_labels,
        "classification_report": report,
        "confusion_matrix": cm,
    }