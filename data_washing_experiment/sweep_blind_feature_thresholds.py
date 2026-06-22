import argparse
import csv
import json
import os

import numpy as np


def load_rows(path):
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed = {}
            for key, value in row.items():
                if key in {"dataset_index", "label", "pred_label", "nearest_label", "ground_truth_poison", "predicted_poison"}:
                    parsed[key] = int(float(value))
                else:
                    parsed[key] = float(value)
            rows.append(parsed)
    return rows


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
    raise ValueError(score_mode)


def evaluate(clean_rows, suspicious_rows, score_mode, quantile):
    clean_scores = np.asarray([build_score(r, score_mode) for r in clean_rows], dtype=np.float32)
    threshold = float(np.quantile(clean_scores, quantile))
    tp = fp = tn = fn = 0
    for row in suspicious_rows:
        pred = int(build_score(row, score_mode) >= threshold)
        gt = int(row["ground_truth_poison"])
        if pred and gt:
            tp += 1
        elif pred and not gt:
            fp += 1
        elif (not pred) and (not gt):
            tn += 1
        else:
            fn += 1
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    accuracy = (tp + tn) / max(1, tp + fp + tn + fn)
    f1 = 0.0 if precision + recall == 0 else (2 * precision * recall / (precision + recall))
    return {
        "score_mode": score_mode,
        "threshold_quantile": quantile,
        "threshold": threshold,
        "accuracy": 100.0 * accuracy,
        "precision": 100.0 * precision,
        "recall": 100.0 * recall,
        "f1": 100.0 * f1,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "missed_poison_rate": 100.0 * fn / max(1, tp + fn),
        "clean_false_positive_rate": 100.0 * fp / max(1, fp + tn),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    args = parser.parse_args()

    clean_rows = load_rows(os.path.join(args.input_dir, "clean_reference_scores.csv"))
    suspicious_rows = load_rows(os.path.join(args.input_dir, "suspicious_scores.csv"))
    score_modes = [
        "pred_centroid_dist",
        "nearest_centroid_dist",
        "pred_dist_plus_uncertainty",
        "nearest_dist_plus_uncertainty",
        "pred_dist_over_margin",
    ]
    quantiles = [0.50, 0.70, 0.80, 0.90, 0.95, 0.99]

    results = []
    for score_mode in score_modes:
        for quantile in quantiles:
            results.append(evaluate(clean_rows, suspicious_rows, score_mode, quantile))

    results.sort(key=lambda row: (-row["f1"], -row["recall"], row["clean_false_positive_rate"]))

    csv_path = os.path.join(args.input_dir, "sweep_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    with open(os.path.join(args.input_dir, "best_sweep_result.json"), "w") as f:
        json.dump(results[0], f, indent=2)

    print(json.dumps({"best": results[0], "csv_path": csv_path}, indent=2))


if __name__ == "__main__":
    main()
