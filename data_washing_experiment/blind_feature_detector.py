import argparse
import csv
import json
import os
import sys
import math
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from data_washing_experiment.clean_reference_detector import (
    CIFAR10_MEAN,
    CIFAR10_STD,
    IndexedSubset,
    SmallCifarClassifier,
    apply_trigger,
    build_trigger_tensors,
    eval_classifier,
    load_index_set,
    parse_client_indices,
    parse_latest_user_lists,
    set_seed,
    train_epoch,
    write_index_file,
)


class MaterializedIndexedSubset(Dataset):
    def __init__(self, dataset, indices, poison_index_set=None, pattern=None, mask=None):
        self.dataset = dataset
        self.indices = [int(index) for index in indices]
        self.poison_index_set = poison_index_set or set()
        self.pattern = pattern
        self.mask = mask

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, position):
        dataset_index = self.indices[position]
        image, label = self.dataset[dataset_index]
        if dataset_index in self.poison_index_set:
            image = apply_trigger(image, self.pattern, self.mask)
        return image, label, dataset_index


def extract_features_and_logits(model, inputs):
    features = model.features(inputs)
    features = torch.flatten(features, 1)
    logits = model.classifier(features)
    return features, logits


def compute_label_centroids(model, loader, device, num_classes):
    sums = [None for _ in range(num_classes)]
    counts = [0 for _ in range(num_classes)]

    model.eval()
    with torch.no_grad():
        for inputs, labels, _ in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            features, _ = extract_features_and_logits(model, inputs)
            for label in labels.unique():
                label_value = int(label.item())
                mask = labels == label
                label_features = features[mask]
                label_sum = label_features.sum(dim=0)
                sums[label_value] = label_sum if sums[label_value] is None else sums[label_value] + label_sum
                counts[label_value] += int(mask.sum().item())

    centroids = []
    for label_value in range(num_classes):
        if counts[label_value] == 0:
            raise ValueError(f"No reference samples found for label {label_value}")
        centroids.append(sums[label_value] / counts[label_value])
    return torch.stack(centroids, dim=0)


def collect_distance_rows(model, loader, centroids, device):
    rows = []
    num_classes = centroids.size(0)
    model.eval()
    with torch.no_grad():
        for inputs, labels, dataset_indices in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            features, logits = extract_features_and_logits(model, inputs)
            probs = torch.softmax(logits, dim=1)
            confidence, pred_labels = probs.max(dim=1)

            distances = torch.cdist(features, centroids)
            pred_dist = distances[torch.arange(distances.size(0), device=device), pred_labels]
            true_dist = distances[torch.arange(distances.size(0), device=device), labels]
            nearest_dist, nearest_label = distances.min(dim=1)

            top2_values, _ = probs.topk(k=min(2, num_classes), dim=1)
            if top2_values.size(1) == 1:
                prob_margin = top2_values[:, 0]
            else:
                prob_margin = top2_values[:, 0] - top2_values[:, 1]

            for index, label, pred_label, conf, p_dist, t_dist, n_dist, n_label, margin in zip(
                dataset_indices.tolist(),
                labels.detach().cpu().tolist(),
                pred_labels.detach().cpu().tolist(),
                confidence.detach().cpu().tolist(),
                pred_dist.detach().cpu().tolist(),
                true_dist.detach().cpu().tolist(),
                nearest_dist.detach().cpu().tolist(),
                nearest_label.detach().cpu().tolist(),
                prob_margin.detach().cpu().tolist(),
            ):
                rows.append(
                    {
                        "dataset_index": int(index),
                        "label": int(label),
                        "pred_label": int(pred_label),
                        "nearest_label": int(n_label),
                        "confidence": float(conf),
                        "pred_centroid_dist": float(p_dist),
                        "true_centroid_dist": float(t_dist),
                        "nearest_centroid_dist": float(n_dist),
                        "prob_margin": float(margin),
                    }
                )
    return rows


def write_rows_csv(path, rows, include_ground_truth=False):
    fieldnames = [
        "dataset_index",
        "label",
        "pred_label",
        "nearest_label",
        "confidence",
        "pred_centroid_dist",
        "true_centroid_dist",
        "nearest_centroid_dist",
        "prob_margin",
    ]
    if rows and "score" in rows[0]:
        fieldnames.append("score")
    if include_ground_truth:
        fieldnames.extend(["ground_truth_poison", "predicted_poison"])
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fit_score_stats(clean_rows, score_mode):
    values = np.asarray([build_score(row, score_mode) for row in clean_rows], dtype=np.float32)
    mean = float(values.mean())
    std = float(values.std())
    if std < 1e-6:
        std = 1.0
    return {"mean": mean, "std": std}


def build_score(row, score_mode):
    if score_mode == "pred_centroid_dist":
        return float(row["pred_centroid_dist"])
    if score_mode == "nearest_centroid_dist":
        return float(row["nearest_centroid_dist"])
    if score_mode == "pred_dist_plus_uncertainty":
        return float(row["pred_centroid_dist"] + (1.0 - row["confidence"]))
    if score_mode == "nearest_dist_plus_uncertainty":
        return float(row["nearest_centroid_dist"] + (1.0 - row["confidence"]))
    if score_mode == "pred_dist_over_margin":
        return float(row["pred_centroid_dist"] / max(1e-6, row["prob_margin"]))
    raise ValueError(f"Unsupported score_mode: {score_mode}")


def percentile_threshold(clean_rows, quantile, score_mode):
    values = np.asarray([build_score(row, score_mode) for row in clean_rows], dtype=np.float32)
    return float(np.quantile(values, quantile))


def evaluate_rows(rows, poison_index_set, threshold, score_mode):
    tp = fp = tn = fn = 0
    predicted_poison_indices = []
    predicted_clean_indices = []
    detailed_rows = []

    for row in rows:
        dataset_index = row["dataset_index"]
        ground_truth_poison = int(dataset_index in poison_index_set)
        score_value = build_score(row, score_mode)
        predicted_poison = int(score_value >= threshold)

        if predicted_poison:
            predicted_poison_indices.append(dataset_index)
        else:
            predicted_clean_indices.append(dataset_index)

        if predicted_poison and ground_truth_poison:
            tp += 1
        elif predicted_poison and not ground_truth_poison:
            fp += 1
        elif (not predicted_poison) and (not ground_truth_poison):
            tn += 1
        else:
            fn += 1

        updated = dict(row)
        updated["ground_truth_poison"] = ground_truth_poison
        updated["score"] = float(score_value)
        updated["predicted_poison"] = int(predicted_poison)
        detailed_rows.append(updated)

    accuracy = (tp + tn) / max(1, tp + fp + tn + fn)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 0.0 if precision + recall == 0 else (2 * precision * recall / (precision + recall))
    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "accuracy": 100.0 * accuracy,
        "precision": 100.0 * precision,
        "recall": 100.0 * recall,
        "f1": 100.0 * f1,
        "predicted_poison_indices": predicted_poison_indices,
        "predicted_clean_indices": predicted_clean_indices,
        "rows": detailed_rows,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--audit-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--data-path", default=".data/")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--threshold-quantile", type=float, default=0.99)
    parser.add_argument(
        "--score-mode",
        default="pred_dist_plus_uncertainty",
        choices=[
            "pred_centroid_dist",
            "nearest_centroid_dist",
            "pred_dist_plus_uncertainty",
            "nearest_dist_plus_uncertainty",
            "pred_dist_over_margin",
        ],
    )
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    client_indices = parse_client_indices(os.path.join(args.run_dir, "all_client.txt"))
    good_clients, suspicious_clients = parse_latest_user_lists(os.path.join(args.run_dir, "log.txt"))
    poison_index_set = load_index_set(os.path.join(args.audit_dir, "ground_truth_poison_indices.txt"))

    reference_indices = []
    suspicious_indices = []
    for client_id, indices in client_indices.items():
        if client_id in good_clients:
            reference_indices.extend(indices)
        elif client_id in suspicious_clients:
            suspicious_indices.extend(indices)

    reference_indices = sorted(set(reference_indices))
    suspicious_indices = sorted(set(suspicious_indices))

    rng = random.Random(args.seed)
    shuffled_reference = reference_indices[:]
    rng.shuffle(shuffled_reference)
    val_size = max(1, int(len(shuffled_reference) * args.val_ratio))
    val_indices = sorted(shuffled_reference[:val_size])
    train_indices = sorted(shuffled_reference[val_size:])

    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )

    train_base = datasets.CIFAR10(root=args.data_path, train=True, download=True, transform=train_transform)
    eval_base = datasets.CIFAR10(root=args.data_path, train=True, download=True, transform=eval_transform)

    pattern, mask = build_trigger_tensors(device=torch.device("cpu"))

    train_loader = DataLoader(IndexedSubset(train_base, train_indices), batch_size=args.batch_size, shuffle=True, num_workers=0)
    train_eval_loader = DataLoader(IndexedSubset(eval_base, train_indices), batch_size=args.batch_size, shuffle=False, num_workers=0)
    val_loader = DataLoader(IndexedSubset(eval_base, val_indices), batch_size=args.batch_size, shuffle=False, num_workers=0)
    suspicious_loader = DataLoader(
        MaterializedIndexedSubset(
            eval_base,
            suspicious_indices,
            poison_index_set=poison_index_set,
            pattern=pattern,
            mask=mask,
        ),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    model = SmallCifarClassifier(num_classes=10).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    history = []
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = eval_classifier(model, val_loader, criterion, device)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }
        )
        print(
            f"[blind_feature] epoch={epoch} train_loss={train_loss:.4f} train_acc={train_acc:.2f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.2f}"
        )

    centroids = compute_label_centroids(model, train_eval_loader, device, num_classes=10)
    clean_rows = collect_distance_rows(model, val_loader, centroids, device)
    suspicious_rows = collect_distance_rows(model, suspicious_loader, centroids, device)

    threshold = percentile_threshold(clean_rows, args.threshold_quantile, args.score_mode)
    suspicious_metrics = evaluate_rows(suspicious_rows, poison_index_set, threshold, args.score_mode)

    all_clean_kept = sorted(set(reference_indices) | set(suspicious_metrics["predicted_clean_indices"]))
    all_poison_filtered = sorted(set(suspicious_metrics["predicted_poison_indices"]))
    overall_tp = suspicious_metrics["tp"]
    overall_fp = suspicious_metrics["fp"]
    overall_tn = suspicious_metrics["tn"] + len(reference_indices)
    overall_fn = suspicious_metrics["fn"]
    overall_total = overall_tp + overall_fp + overall_tn + overall_fn
    overall_accuracy = 100.0 * (overall_tp + overall_tn) / max(1, overall_total)

    summary = {
        "seed": args.seed,
        "device": str(device),
        "good_clients": good_clients,
        "suspicious_clients": suspicious_clients,
        "reference_train_samples": len(train_indices),
        "reference_val_samples": len(val_indices),
        "suspicious_samples": len(suspicious_indices),
        "threshold_quantile": args.threshold_quantile,
        "threshold": threshold,
        "score_mode": args.score_mode,
        "train_history": history,
        "suspicious_metrics": {
            "accuracy": suspicious_metrics["accuracy"],
            "precision": suspicious_metrics["precision"],
            "recall": suspicious_metrics["recall"],
            "f1": suspicious_metrics["f1"],
            "tp": suspicious_metrics["tp"],
            "fp": suspicious_metrics["fp"],
            "tn": suspicious_metrics["tn"],
            "fn": suspicious_metrics["fn"],
        },
        "overall_cleaned_dataset_accuracy": overall_accuracy,
        "predicted_poison_samples": len(all_poison_filtered),
        "predicted_clean_samples": len(all_clean_kept),
        "missed_poison_samples": suspicious_metrics["fn"],
        "missed_poison_rate": 100.0 * suspicious_metrics["fn"] / max(1, suspicious_metrics["tp"] + suspicious_metrics["fn"]),
        "clean_false_positive_samples": suspicious_metrics["fp"],
        "clean_false_positive_rate_in_suspicious": 100.0 * suspicious_metrics["fp"] / max(1, suspicious_metrics["fp"] + suspicious_metrics["tn"]),
    }

    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    write_rows_csv(os.path.join(args.output_dir, "clean_reference_scores.csv"), clean_rows)
    write_rows_csv(
        os.path.join(args.output_dir, "suspicious_scores.csv"),
        suspicious_metrics["rows"],
        include_ground_truth=True,
    )

    write_index_file(os.path.join(args.output_dir, "predicted_poison_indices.txt"), all_poison_filtered)
    write_index_file(os.path.join(args.output_dir, "predicted_clean_indices.txt"), all_clean_kept)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
