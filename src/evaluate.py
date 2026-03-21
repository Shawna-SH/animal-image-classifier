"""
Evaluate a trained image classification model on validation or test data.

This script loads a trained model checkpoint and evaluates its performance
using the project's existing data pipeline and model architecture. It computes
standard classification metrics and saves results for further analysis.

Usage:
    Run from the project root directory:

    python -m src.evaluate \
        --checkpoint artifacts/models/best_model.pth \
        --output-dir artifacts/evaluation \
        --batch-size 32 \
        --split test

Arguments:
    --checkpoint (str, required):
        Path to the trained model checkpoint (.pth file).

    --output-dir (str, optional):
        Directory to save evaluation outputs.
        Default: artifacts/evaluation

    --data-dir (str, optional):
        Root data directory used by the data loader.
        Default: data/raw

    --image-size (int, optional):
        Input image size for the model.
        Default: 224

    --batch-size (int, optional):
        Batch size for evaluation.
        Default: 32

    --num-workers (int, optional):
        Number of DataLoader workers.
        Default: 0

    --split (str, optional):
        Dataset split to evaluate on: "val" or "test".
        Default: test

    --device (str, optional):
        Device to use: "auto", "cpu", or "cuda".
        Default: auto

Outputs:
    The following files will be saved to --output-dir:

    - {split}_report.txt
        Classification report (precision, recall, F1-score per class)

    - {split}_confusion_matrix.png
        Confusion matrix visualization

    - {split}_y_true.npy
        Ground truth labels

    - {split}_y_pred.npy
        Predicted labels

    - {split}_y_prob.npy
        Predicted probabilities

Notes:
    - This script reuses the project's data_loader and model modules.
    - The model architecture must match the checkpoint.
    - The number of classes is inferred from the dataset.

Example:
    python -m src.evaluate \
        --checkpoint artifacts/models/best_model.pth \
        --split val
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from src.data_loader import create_dataloaders
from src.model import build_model


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trained animal classifier.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint (.pth)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="artifacts/evaluation",
        help="Directory to save evaluation outputs",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/raw",
        help="Root data directory used by data_loader.py",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        help="Input image size",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of dataloader workers",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["val", "test"],
        help="Which split to evaluate on",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to use",
    )
    return parser.parse_args()


def get_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA requested but not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_checkpoint(model: torch.nn.Module, checkpoint_path: str, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    return model


def evaluate(model, dataloader, device):
    model.eval()

    y_true = []
    y_pred = []
    y_prob = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_prob.extend(probs.cpu().numpy())

    return np.array(y_true), np.array(y_pred), np.array(y_prob)


def save_confusion_matrix(cm, class_names, save_path: Path):
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.colorbar()

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha="right")
    plt.yticks(tick_marks, class_names)

    threshold = cm.max() / 2 if cm.max() > 0 else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > threshold else "black",
            )

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def main():
    args = parse_args()
    device = get_device(args.device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, test_loader, class_names = create_dataloaders(
        data_dir=args.data_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    dataloader = val_loader if args.split == "val" else test_loader
    num_classes = len(class_names)

    model = build_model(num_classes=num_classes, pretrained=False)
    model = load_checkpoint(model, args.checkpoint, device)
    model.to(device)

    print(f"Using device: {device}")
    print(f"Evaluating split: {args.split}")
    print(f"Classes: {class_names}")

    y_true, y_pred, y_prob = evaluate(model, dataloader, device)

    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    report_text = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        zero_division=0,
    )

    with open(output_dir / f"{args.split}_report.txt", "w", encoding="utf-8") as f:
        f.write(report_text)

    np.save(output_dir / f"{args.split}_y_true.npy", y_true)
    np.save(output_dir / f"{args.split}_y_pred.npy", y_pred)
    np.save(output_dir / f"{args.split}_y_prob.npy", y_prob)

    save_confusion_matrix(cm, class_names, output_dir / f"{args.split}_confusion_matrix.png")

    print(f"\nAccuracy: {acc:.4f}\n")
    print(report_text)
    print(f"Saved outputs to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()