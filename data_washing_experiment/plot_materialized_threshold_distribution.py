import argparse
import csv
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, transforms

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from data_washing_experiment.clean_reference_detector import (
    CIFAR10_MEAN,
    CIFAR10_STD,
    parse_client_indices,
    parse_latest_user_lists,
)
from data_washing_experiment.materialized_trigger_detector import (
    build_trigger_tensors,
    masked_trigger_mse,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--data-path", default=".data/")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.summary_json, "r") as f:
        summary = json.load(f)

    threshold = float(summary["threshold"])
    threshold_quantile = float(summary["threshold_quantile"])

    client_indices = parse_client_indices(os.path.join(args.run_dir, "all_client.txt"))
    good_clients, _ = parse_latest_user_lists(os.path.join(args.run_dir, "log.txt"))

    eval_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )
    dataset = datasets.CIFAR10(root=args.data_path, train=True, download=True, transform=eval_transform)
    pattern, mask = build_trigger_tensors()

    rows = []
    for client_id in good_clients:
        for dataset_index in client_indices.get(client_id, []):
            input_tensor, label = dataset[int(dataset_index)]
            patch_mse = masked_trigger_mse(input_tensor, pattern, mask)
            rows.append(
                {
                    "client_id": int(client_id),
                    "dataset_index": int(dataset_index),
                    "label": int(label),
                    "patch_mse": float(patch_mse),
                }
            )

    scores = np.asarray([row["patch_mse"] for row in rows], dtype=np.float32)
    quantile_value = float(np.quantile(scores, threshold_quantile))

    csv_path = os.path.join(args.output_dir, "clean_reference_patch_mse.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["client_id", "dataset_index", "label", "patch_mse"])
        writer.writeheader()
        writer.writerows(rows)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(scores, bins=100, color="#4C78A8", alpha=0.85, edgecolor="white")
    ax.axvline(
        threshold,
        color="#E45756",
        linestyle="--",
        linewidth=2,
        label=f"threshold = {threshold:.6f}",
    )
    ax.axvline(
        quantile_value,
        color="#72B7B2",
        linestyle=":",
        linewidth=2,
        label=f"q={threshold_quantile:.2f} = {quantile_value:.6f}",
    )
    ax.set_title("Clean Reference Patch-MSE Distribution")
    ax.set_xlabel("patch_mse")
    ax.set_ylabel("count")
    ax.legend()
    ax.grid(alpha=0.2)

    png_path = os.path.join(args.output_dir, "clean_reference_patch_mse_distribution.png")
    fig.tight_layout()
    fig.savefig(png_path, dpi=180)
    plt.close(fig)

    stats = {
        "good_clients": good_clients,
        "sample_count": int(scores.size),
        "threshold": threshold,
        "threshold_quantile": threshold_quantile,
        "min": float(scores.min()),
        "max": float(scores.max()),
        "mean": float(scores.mean()),
        "std": float(scores.std()),
        "median": float(np.median(scores)),
        "quantile_value_recomputed": quantile_value,
        "csv_path": csv_path,
        "png_path": png_path,
    }
    with open(os.path.join(args.output_dir, "clean_reference_patch_mse_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
