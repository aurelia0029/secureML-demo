import argparse
import csv
import json
import os
import sys
import time

import numpy as np
import torch
from torchvision import datasets, transforms

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from data_washing_experiment.clean_reference_detector import (
    BACKDOOR_PATTERN,
    BACKDOOR_X_TOP,
    BACKDOOR_Y_TOP,
    CIFAR10_MEAN,
    CIFAR10_STD,
    MASK_VALUE,
    load_index_set,
    parse_client_indices,
    parse_latest_user_lists,
)


def build_trigger_tensors():
    full_image = torch.zeros((3, 32, 32), dtype=torch.float32)
    full_image.fill_(MASK_VALUE)
    x_bot = BACKDOOR_X_TOP + BACKDOOR_PATTERN.shape[0]
    y_bot = BACKDOOR_Y_TOP + BACKDOOR_PATTERN.shape[1]
    full_image[:, BACKDOOR_X_TOP:x_bot, BACKDOOR_Y_TOP:y_bot] = BACKDOOR_PATTERN
    mask = (full_image != MASK_VALUE).to(torch.float32)
    pattern = transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)(full_image)
    return pattern, mask


def build_trigger_patch_arrays():
    pattern, mask = build_trigger_tensors()
    x_bot = BACKDOOR_X_TOP + BACKDOOR_PATTERN.shape[0]
    y_bot = BACKDOOR_Y_TOP + BACKDOOR_PATTERN.shape[1]
    pattern_patch = pattern[:, BACKDOOR_X_TOP:x_bot, BACKDOOR_Y_TOP:y_bot].numpy()
    mask_patch = mask[:, BACKDOOR_X_TOP:x_bot, BACKDOOR_Y_TOP:y_bot].numpy().astype(bool)
    return pattern_patch, mask_patch


def apply_trigger(input_tensor, pattern, mask):
    return (1 - mask) * input_tensor + mask * pattern


def masked_trigger_mse(input_tensor, pattern, mask):
    diff = (input_tensor - pattern) * mask
    denom = mask.sum().item()
    if denom <= 0:
        return 0.0
    return float((diff.pow(2).sum() / denom).item())


def percentile_threshold(scores, quantile):
    scores = np.asarray(scores, dtype=np.float32)
    return float(np.quantile(scores, quantile))


def normalized_patch_from_uint8(image_hwc, mean, std):
    patch = image_hwc[
        BACKDOOR_X_TOP:BACKDOOR_X_TOP + BACKDOOR_PATTERN.shape[0],
        BACKDOOR_Y_TOP:BACKDOOR_Y_TOP + BACKDOOR_PATTERN.shape[1],
        :
    ].astype(np.float32) / 255.0
    patch = np.transpose(patch, (2, 0, 1))
    return (patch - mean[:, None, None]) / std[:, None, None]


def compute_patch_mse_rows(client_ids, client_indices, poison_index_set, dataset_data, dataset_labels, pattern_patch, mask_patch):
    mean = np.asarray(CIFAR10_MEAN, dtype=np.float32)
    std = np.asarray(CIFAR10_STD, dtype=np.float32)
    denom = float(mask_patch.sum())
    rows = []
    total = len(client_ids)
    start = time.time()

    for pos, client_id in enumerate(client_ids, start=1):
        for dataset_index in client_indices.get(client_id, []):
            image = dataset_data[int(dataset_index)]
            ground_truth_poison = int(dataset_index in poison_index_set)
            if ground_truth_poison:
                patch_mse = 0.0
            else:
                normalized_patch = normalized_patch_from_uint8(image, mean, std)
                diff = (normalized_patch - pattern_patch)[mask_patch]
                patch_mse = float(np.dot(diff, diff) / max(1.0, denom))

            rows.append(
                {
                    "client_id": int(client_id),
                    "dataset_index": int(dataset_index),
                    "label": int(dataset_labels[int(dataset_index)]),
                    "ground_truth_poison": ground_truth_poison,
                    "patch_mse": patch_mse,
                }
            )

        elapsed = time.time() - start
        print(
            f"[materialized] clients={pos}/{total} ({100.0 * pos / max(1, total):.1f}%) "
            f"rows={len(rows)} elapsed={elapsed:.1f}s"
        , flush=True)

    return rows


def evaluate_rows(rows, threshold):
    tp = fp = tn = fn = 0
    predicted_poison_indices = []
    predicted_clean_indices = []

    for row in rows:
        predicted_poison = int(row["patch_mse"] <= threshold)
        row["predicted_poison"] = predicted_poison
        if predicted_poison:
            predicted_poison_indices.append(row["dataset_index"])
        else:
            predicted_clean_indices.append(row["dataset_index"])

        gt = row["ground_truth_poison"]
        if predicted_poison and gt:
            tp += 1
        elif predicted_poison and not gt:
            fp += 1
        elif (not predicted_poison) and (not gt):
            tn += 1
        else:
            fn += 1

    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    accuracy = (tp + tn) / max(1, tp + fp + tn + fn)
    f1 = 0.0 if precision + recall == 0 else (2 * precision * recall / (precision + recall))
    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": 100.0 * precision,
        "recall": 100.0 * recall,
        "accuracy": 100.0 * accuracy,
        "f1": 100.0 * f1,
        "predicted_poison_indices": predicted_poison_indices,
        "predicted_clean_indices": predicted_clean_indices,
    }


def write_index_file(path, values):
    with open(path, "w") as f:
        for value in values:
            f.write(f"{int(value)}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--audit-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--data-path", default=os.path.join(ROOT_DIR, ".data"))
    parser.add_argument("--threshold-quantile", type=float, default=0.01)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    client_indices = parse_client_indices(os.path.join(args.run_dir, "all_client.txt"))
    good_clients, suspicious_clients = parse_latest_user_lists(os.path.join(args.run_dir, "log.txt"))
    poison_index_set = load_index_set(os.path.join(args.audit_dir, "ground_truth_poison_indices.txt"))

    dataset = datasets.CIFAR10(root=args.data_path, train=True, download=True)
    dataset_data = dataset.data
    dataset_labels = dataset.targets
    pattern_patch, mask_patch = build_trigger_patch_arrays()

    clean_rows = compute_patch_mse_rows(
        good_clients,
        client_indices,
        set(),
        dataset_data,
        dataset_labels,
        pattern_patch,
        mask_patch,
    )
    clean_reference_scores = [row["patch_mse"] for row in clean_rows]
    suspicious_rows = compute_patch_mse_rows(
        suspicious_clients,
        client_indices,
        poison_index_set,
        dataset_data,
        dataset_labels,
        pattern_patch,
        mask_patch,
    )

    threshold = percentile_threshold(clean_reference_scores, args.threshold_quantile)
    metrics = evaluate_rows(suspicious_rows, threshold)

    summary = {
        "good_clients": good_clients,
        "suspicious_clients": suspicious_clients,
        "threshold_quantile": args.threshold_quantile,
        "threshold": threshold,
        "reference_clean_samples": len(clean_reference_scores),
        "suspicious_samples": len(suspicious_rows),
        "suspicious_metrics": {
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "tp": metrics["tp"],
            "fp": metrics["fp"],
            "tn": metrics["tn"],
            "fn": metrics["fn"],
        },
        "missed_poison_samples": metrics["fn"],
        "missed_poison_rate": 100.0 * metrics["fn"] / max(1, metrics["tp"] + metrics["fn"]),
        "clean_false_positive_samples": metrics["fp"],
        "clean_false_positive_rate_in_suspicious": 100.0 * metrics["fp"] / max(1, metrics["fp"] + metrics["tn"]),
    }

    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    with open(os.path.join(args.output_dir, "suspicious_patch_scores.csv"), "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "client_id",
                "dataset_index",
                "label",
                "ground_truth_poison",
                "patch_mse",
                "predicted_poison",
            ],
        )
        writer.writeheader()
        writer.writerows(suspicious_rows)

    write_index_file(os.path.join(args.output_dir, "predicted_poison_indices.txt"), metrics["predicted_poison_indices"])
    write_index_file(os.path.join(args.output_dir, "predicted_clean_indices.txt"), metrics["predicted_clean_indices"])
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
