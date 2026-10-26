#!/usr/bin/env python3
"""
Compare class-imbalance strategies for EfficientNet-B0 on CSV image splits

Strategies Tested
----------
  plain_ce                  Standard cross-entropy (baseline), uniform shuffling.
  weighted_ce               Cross-entropy with inverse-frequency class weights.
  weighted_ce_oversampling  Weighted cross-entropy + WeightedRandomSampler
                            (minority classes are drawn more often).

CSV format (train / val / test)
-------------------------------
  lesion_id, image_id, dx, dx_type, age, sex, localization, dataset, image_path

  `dx` is the label column and `image_path` the image file (absolute, or relative
  to --image-root). Both column names can be changed with --label-col / --path-col.

Example
-------
  python loss_tests.py \
      --train-csv data/train.csv --val-csv data/val.csv --test-csv data/test.csv \
      --strategy weighted_ce_oversampling --epochs 20 --batch-size 32

Every run writes to  <output-dir>/<strategy>_<timestamp>/ :
  run_config.json     strategy, all CLI parameters, class counts, class weights,
                      per-class sampling weights/probabilities, split statistics
  history.csv         per-epoch metrics (+ classes actually drawn when oversampling)
  best_model.pth      best checkpoint by validation macro-F1
  test_results.json   final test metrics and per-class report
  test_predictions.csv  per-image predictions on the test set
"""

import argparse
import contextlib
import csv
import json
import os
import random
import sys
import time
from collections import Counter
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models, transforms
from torchvision.models import EfficientNet_B0_Weights

STRATEGIES = ("plain_ce", "weighted_ce", "weighted_ce_oversampling")
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ---------------------------------------------------------------------------
# Command-line arguments
#


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="EfficientNet-B0 loss / imbalance-strategy experiments.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    data = p.add_argument_group("data")
    data.add_argument("--train-csv", required=True, help="Training split CSV.")
    data.add_argument("--val-csv", required=True, help="Validation split CSV.")
    data.add_argument("--test-csv", required=True, help="Test split CSV.")
    data.add_argument(
        "--image-root",
        default="",
        help="Prefix for relative paths in the image_path column.",
    )
    data.add_argument("--label-col", default="dx", help="Label column name.")
    data.add_argument(
        "--path-col", default="image_path", help="Image path column name."
    )
    data.add_argument("--img-size", type=int, default=224, help="Input resolution.")

    strat = p.add_argument_group("imbalance strategy")
    strat.add_argument(
        "--strategy",
        choices=STRATEGIES,
        default="plain_ce",
        help="Loss / sampling strategy.",
    )
    strat.add_argument(
        "--weight-power",
        type=float,
        default=1.0,
        help="Class weight = (N / (K * n_c)) ** power. "
        "1.0 = inverse frequency, 0.5 = inverse sqrt.",
    )
    strat.add_argument(
        "--sampler-power",
        type=float,
        default=1.0,
        help="Oversampling: per-sample weight = (1 / n_c) ** power. "
        "1.0 = class-balanced draws, 0.0 = natural distribution.",
    )
    strat.add_argument(
        "--samples-per-epoch",
        type=int,
        default=0,
        help="Oversampling: images drawn per epoch (0 = size of train set).",
    )
    strat.add_argument(
        "--no-replacement",
        action="store_true",
        help="Oversampling: draw without replacement.",
    )
    strat.add_argument(
        "--label-smoothing",
        type=float,
        default=0.0,
        help="Label smoothing for cross-entropy.",
    )

    train = p.add_argument_group("training")
    train.add_argument("--epochs", type=int, default=20)
    train.add_argument("--batch-size", type=int, default=32)
    train.add_argument("--lr", type=float, default=1e-3)
    train.add_argument("--weight-decay", type=float, default=1e-4)
    train.add_argument(
        "--lr-patience",
        type=int,
        default=2,
        help="ReduceLROnPlateau patience (on val macro-F1).",
    )
    train.add_argument("--lr-factor", type=float, default=0.5)
    train.add_argument(
        "--early-stop",
        type=int,
        default=0,
        help="Stop after N epochs without val macro-F1 gain (0 = off).",
    )
    train.add_argument("--num-workers", type=int, default=4)
    train.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Train from scratch instead of ImageNet weights.",
    )
    train.add_argument(
        "--amp", action="store_true", help="Mixed precision (CUDA only)."
    )
    train.add_argument("--seed", type=int, default=42)
    train.add_argument(
        "--device", default="auto", help="auto, cpu, cuda, cuda:0, mps ..."
    )

    out = p.add_argument_group("output")
    out.add_argument(
        "--output-dir", default="runs", help="Parent folder for run outputs."
    )
    out.add_argument(
        "--run-name",
        default="",
        help="Run folder name (default: <strategy>_<timestamp>).",
    )

    args = p.parse_args(argv)

    if args.epochs < 1 or args.batch_size < 1:
        p.error("--epochs and --batch-size must be >= 1")
    if args.strategy != "weighted_ce_oversampling" and (
        args.samples_per_epoch or args.no_replacement or args.sampler_power != 1.0
    ):
        print(
            "[warn] Sampler options are ignored unless --strategy "
            "weighted_ce_oversampling.",
            file=sys.stderr,
        )
    return args


# ---------------------------------------------------------------------------
# Utilities
#


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(name):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_split(csv_path, args, split):
    if not os.path.isfile(csv_path):
        sys.exit(f"[error] {split} CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    # Strip whitespace and invisible characters (e.g. zero-width spaces) from headers.
    df.columns = [c.strip().strip("​﻿") for c in df.columns]

    for col in (args.label_col, args.path_col):
        if col not in df.columns:
            sys.exit(
                f"[error] {split} CSV is missing column '{col}'. "
                f"Found: {list(df.columns)}"
            )

    before = len(df)
    df = df.dropna(subset=[args.label_col, args.path_col]).reset_index(drop=True)
    if len(df) < before:
        print(
            f"[warn] {split}: dropped {before - len(df)} rows with missing label/path."
        )

    df[args.label_col] = df[args.label_col].astype(str).str.strip()
    df["_full_path"] = [
        p
        if os.path.isabs(p) or not args.image_root
        else os.path.join(args.image_root, p)
        for p in df[args.path_col].astype(str).str.strip()
    ]

    missing = [p for p in df["_full_path"] if not os.path.isfile(p)]
    if missing:
        sys.exit(
            f"[error] {split}: {len(missing)} image files not found, "
            f"e.g. {missing[:3]}. Check --image-root / --path-col."
        )
    if len(df) == 0:
        sys.exit(f"[error] {split} split is empty.")
    return df


def check_lesion_leakage(splits):
    if not all("lesion_id" in df.columns for df in splits.values()):
        return {}
    ids = {k: set(df["lesion_id"].dropna().astype(str)) for k, df in splits.items()}
    overlap = {
        "train_val": len(ids["train"] & ids["val"]),
        "train_test": len(ids["train"] & ids["test"]),
        "val_test": len(ids["val"] & ids["test"]),
    }
    if any(overlap.values()):
        print(f"[warn] lesion_id overlap between splits (possible leakage): {overlap}")
    return overlap


class CSVImageDataset(Dataset):
    def __init__(self, df, label_col, class_to_idx, transform):
        self.paths = df["_full_path"].tolist()
        self.image_ids = (
            df["image_id"].astype(str).tolist()
            if "image_id" in df.columns
            else self.paths
        )
        self.labels = [class_to_idx[c] for c in df[label_col]]
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with Image.open(self.paths[i]) as img:
            img = img.convert("RGB")
        return self.transform(img), self.labels[i], i


# ---------------------------------------------------------------------------
# Training and evaluation
#


def train_one_epoch(
    model, loader, criterion, optimizer, scaler, device, num_classes, amp
):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    drawn = torch.zeros(num_classes, dtype=torch.long)  # class mix actually seen

    for images, labels, _ in loader:
        drawn += torch.bincount(labels, minlength=num_classes)
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda") if amp else contextlib.nullcontext():
            logits = model(images)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * images.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total, drawn.tolist()


@torch.no_grad()
def evaluate(model, loader, criterion, device, class_names, amp):
    model.eval()
    total_loss, total = 0.0, 0
    y_true, y_pred, probs_all, idx_all = [], [], [], []

    for images, labels, idx in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.autocast("cuda") if amp else contextlib.nullcontext():
            logits = model(images)
            loss = criterion(logits, labels)

        total_loss += loss.item() * images.size(0)
        total += labels.size(0)
        y_true += labels.cpu().tolist()
        y_pred += logits.argmax(1).cpu().tolist()
        probs_all.append(torch.softmax(logits.float(), 1).cpu())
        idx_all += idx.tolist()

    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    return {
        "loss": total_loss / total,
        "accuracy": report["accuracy"],
        "macro_f1": report["macro avg"]["f1-score"],
        "macro_recall": report["macro avg"]["recall"],
        "weighted_f1": report["weighted avg"]["f1-score"],
        "report": report,
        "y_true": y_true,
        "y_pred": y_pred,
        "probs": torch.cat(probs_all).numpy(),
        "indices": idx_all,
    }


# ---------------------------------------------------------------------------
# Main
#


def main(argv=None):
    args = parse_args(argv)
    set_seed(args.seed)
    device = resolve_device(args.device)
    amp = bool(args.amp and device.type == "cuda")
    pin = device.type == "cuda"

    run_name = args.run_name or f"{args.strategy}_{datetime.now():%Y%m%d_%H%M%S}"
    run_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    print(f"Device: {device} | Strategy: {args.strategy} | Run dir: {run_dir}")

    # load data splits
    splits = {
        "train": load_split(args.train_csv, args, "train"),
        "val": load_split(args.val_csv, args, "val"),
        "test": load_split(args.test_csv, args, "test"),
    }
    leakage = check_lesion_leakage(splits)

    # class index mapping comes from the training split
    class_names = sorted(splits["train"][args.label_col].unique())
    class_to_idx = {c: i for i, c in enumerate(class_names)}
    num_classes = len(class_names)
    for split in ("val", "test"):
        unseen = set(splits[split][args.label_col]) - set(class_names)
        if unseen:
            sys.exit(
                f"[error] {split} has labels not present in train: {sorted(unseen)}"
            )

    train_tf = transforms.Compose(
        [
            transforms.RandomResizedCrop(args.img_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),  # lesion images have no specific orientation
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    eval_tf = transforms.Compose(
        [
            transforms.Resize(int(args.img_size * 256 / 224)),
            transforms.CenterCrop(args.img_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )

    train_ds = CSVImageDataset(splits["train"], args.label_col, class_to_idx, train_tf)
    val_ds = CSVImageDataset(splits["val"], args.label_col, class_to_idx, eval_tf)
    test_ds = CSVImageDataset(splits["test"], args.label_col, class_to_idx, eval_tf)

    # ---- Class statistics and weights ------------------------------------
    counts_map = Counter(train_ds.labels)
    counts = torch.tensor(
        [counts_map[i] for i in range(num_classes)], dtype=torch.float64
    )
    n_train = int(counts.sum())

    # Inverse-frequency weights (normalised so a balanced dataset gives 1.0).
    class_weights = (counts.sum() / (num_classes * counts)) ** args.weight_power
    class_weights = class_weights / class_weights.mean()  # mean weight = 1

    sampler_info = None
    if args.strategy == "weighted_ce_oversampling":
        per_class_w = (1.0 / counts) ** args.sampler_power
        sample_weights = torch.tensor(
            [per_class_w[y] for y in train_ds.labels], dtype=torch.double
        )
        num_samples = args.samples_per_epoch or n_train
        replacement = not args.no_replacement
        if not replacement and num_samples > n_train:
            sys.exit(
                "[error] --samples-per-epoch exceeds train size without replacement."
            )
        sampler = WeightedRandomSampler(
            sample_weights,
            num_samples=num_samples,
            replacement=replacement,
            generator=torch.Generator().manual_seed(args.seed),
        )
        class_prob = per_class_w * counts / (per_class_w * counts).sum()
        sampler_info = {
            "type": "WeightedRandomSampler",
            "sample_weight_formula": f"(1 / n_class) ** {args.sampler_power}",
            "num_samples_per_epoch": num_samples,
            "replacement": replacement,
            "per_class_sample_weight": {
                c: float(per_class_w[i]) for i, c in enumerate(class_names)
            },
            "expected_draw_probability": {
                c: float(class_prob[i]) for i, c in enumerate(class_names)
            },
            "expected_draws_per_epoch": {
                c: float(class_prob[i] * num_samples) for i, c in enumerate(class_names)
            },
        }
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            sampler=sampler,
            num_workers=args.num_workers,
            pin_memory=pin,
        )
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=pin,
            generator=torch.Generator().manual_seed(args.seed),
        )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin,
    )

    use_weights = args.strategy in ("weighted_ce", "weighted_ce_oversampling")
    criterion = nn.CrossEntropyLoss(
        weight=class_weights.float().to(device) if use_weights else None,
        label_smoothing=args.label_smoothing,
    )

    # record the run configuration
    def dist(df):
        return (
            df[args.label_col]
            .value_counts()
            .reindex(class_names, fill_value=0)
            .astype(int)
            .to_dict()
        )

    run_config = {
        "run_name": run_name,
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "strategy": args.strategy,
        "loss": {
            "type": "CrossEntropyLoss",
            "class_weighted": use_weights,
            "weight_formula": (
                f"(N / (K * n_class)) ** {args.weight_power}, normalised to mean 1"
                if use_weights
                else None
            ),
            "class_weights": (
                {c: float(class_weights[i]) for i, c in enumerate(class_names)}
                if use_weights
                else None
            ),
            "label_smoothing": args.label_smoothing,
        },
        "training_set_selection": {
            "train_csv": os.path.abspath(args.train_csv),
            "num_train_images": n_train,
            "class_counts": {c: int(counts[i]) for i, c in enumerate(class_names)},
            "sampling": sampler_info
            or {
                "type": "uniform shuffle",
                "replacement": False,
                "num_samples_per_epoch": n_train,
            },
        },
        "splits": {
            "val_csv": os.path.abspath(args.val_csv),
            "test_csv": os.path.abspath(args.test_csv),
            "sizes": {k: len(v) for k, v in splits.items()},
            "class_distribution": {k: dist(v) for k, v in splits.items()},
            "lesion_id_overlap": leakage,
        },
        "class_to_idx": class_to_idx,
        "model": "efficientnet_b0",
        "pretrained": not args.no_pretrained,
        "device": str(device),
        "amp": amp,
        "torch_version": torch.__version__,
        "command": " ".join([os.path.basename(sys.argv[0])] + (argv or sys.argv[1:])),
        "args": vars(args),
    }
    with open(os.path.join(run_dir, "run_config.json"), "w") as f:
        json.dump(run_config, f, indent=2)

    print("Classes:", class_names)
    print("Train counts:", run_config["training_set_selection"]["class_counts"])
    if use_weights:
        print(
            "Class weights:",
            {k: round(v, 4) for k, v in run_config["loss"]["class_weights"].items()},
        )
    if sampler_info:
        print(
            "Expected draw probability:",
            {
                k: round(v, 4)
                for k, v in sampler_info["expected_draw_probability"].items()
            },
        )

    # Model Loading -----------------------------------------------------------
    weights = None if args.no_pretrained else EfficientNet_B0_Weights.DEFAULT
    model = models.efficientnet_b0(weights=weights)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    model = model.to(device)

    optimizer = optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=args.lr_factor, patience=args.lr_patience
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    # Training loop -----------------------------------------------------------
    ckpt_path = os.path.join(run_dir, "best_model.pth")
    history_path = os.path.join(run_dir, "history.csv")
    best_f1, best_epoch, stale = -1.0, 0, 0

    with open(history_path, "w", newline="") as hf:
        writer = csv.writer(hf)
        writer.writerow(
            [
                "epoch",
                "lr",
                "train_loss",
                "train_acc",
                "val_loss",
                "val_acc",
                "val_macro_f1",
                "val_macro_recall",
                "seconds",
            ]
            + [f"drawn_{c}" for c in class_names]
        )

        for epoch in range(1, args.epochs + 1):
            t0 = time.time()
            lr = optimizer.param_groups[0]["lr"]
            tr_loss, tr_acc, drawn = train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                scaler,
                device,
                num_classes,
                amp,
            )
            val = evaluate(model, val_loader, criterion, device, class_names, amp)
            scheduler.step(val["macro_f1"])
            secs = time.time() - t0

            writer.writerow(
                [
                    epoch,
                    lr,
                    tr_loss,
                    tr_acc,
                    val["loss"],
                    val["accuracy"],
                    val["macro_f1"],
                    val["macro_recall"],
                    round(secs, 1),
                ]
                + drawn
            )
            hf.flush()

            print(
                f"Epoch {epoch}/{args.epochs} | lr {lr:.2e} | "
                f"train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
                f"val loss {val['loss']:.4f} acc {val['accuracy']:.4f} "
                f"macro-F1 {val['macro_f1']:.4f} macro-recall {val['macro_recall']:.4f} | "
                f"{secs:.0f}s"
            )

            if val["macro_f1"] > best_f1:
                best_f1, best_epoch, stale = val["macro_f1"], epoch, 0
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "class_to_idx": class_to_idx,
                        "class_weights": class_weights.float(),
                        "strategy": args.strategy,
                        "val_macro_f1": val["macro_f1"],
                    },
                    ckpt_path,
                )
                print("  -> saved best checkpoint")
            else:
                stale += 1
                if args.early_stop and stale >= args.early_stop:
                    print(f"Early stopping: no val macro-F1 gain for {stale} epochs.")
                    break

    # Final test results/report -----------------------------------------------
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    test = evaluate(model, test_loader, criterion, device, class_names, amp)

    print(f"\nFinal test results (best epoch {best_epoch}, val macro-F1 {best_f1:.4f})")
    print(f"Strategy:        {args.strategy}")
    print(f"Test loss:       {test['loss']:.4f}")
    print(f"Test accuracy:   {test['accuracy']:.4f}")
    print(f"Test macro-F1:   {test['macro_f1']:.4f}")
    print(f"Test macro-rec.: {test['macro_recall']:.4f}")
    print("\nPer-class results:")
    for c in class_names:
        m = test["report"][c]
        print(
            f"  {c:>10}: precision={m['precision']:.4f} recall={m['recall']:.4f} "
            f"F1={m['f1-score']:.4f} support={int(m['support'])}"
        )

    cm = confusion_matrix(
        test["y_true"], test["y_pred"], labels=list(range(num_classes))
    )
    with open(os.path.join(run_dir, "test_results.json"), "w") as f:
        json.dump(
            {
                "strategy": args.strategy,
                "best_epoch": best_epoch,
                "best_val_macro_f1": best_f1,
                "test_loss": test["loss"],
                "test_accuracy": test["accuracy"],
                "test_macro_f1": test["macro_f1"],
                "test_macro_recall": test["macro_recall"],
                "test_weighted_f1": test["weighted_f1"],
                "per_class": {c: test["report"][c] for c in class_names},
                "confusion_matrix": {"labels": class_names, "matrix": cm.tolist()},
            },
            f,
            indent=2,
        )

    with open(os.path.join(run_dir, "test_predictions.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_id", "true", "pred"] + [f"prob_{c}" for c in class_names])
        for row, i in enumerate(test["indices"]):
            w.writerow(
                [
                    test_ds.image_ids[i],
                    class_names[test["y_true"][row]],
                    class_names[test["y_pred"][row]],
                ]
                + [f"{p:.6f}" for p in test["probs"][row]]
            )

    print(f"\nOutputs saved to {run_dir}")


if __name__ == "__main__":
    main()
