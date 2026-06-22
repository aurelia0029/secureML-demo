import argparse
import csv
import json
import os
import random
import re
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)
BACKDOOR_PATTERN = torch.tensor(
    [
        [1.0, 0.0, 1.0],
        [-10.0, 1.0, -10.0],
        [-10.0, -10.0, 0.0],
        [-10.0, 1.0, -10.0],
        [1.0, 0.0, 1.0],
    ],
    dtype=torch.float32,
)
BACKDOOR_X_TOP = 3
BACKDOOR_Y_TOP = 23
MASK_VALUE = -10.0


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class IndexedSubset(Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = [int(index) for index in indices]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, position):
        dataset_index = self.indices[position]
        image, label = self.dataset[dataset_index]
        return image, label, dataset_index


class SmallCifarClassifier(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


@dataclass
class DetectorPaths:
    run_dir: str
    audit_dir: str
    output_dir: str


def parse_client_indices(all_client_path):
    clients = {}
    current_client = None
    with open(all_client_path) as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("id:"):
                current_client = int(line.split(":", 1)[1].strip())
                clients[current_client] = []
                continue
            if line.startswith("-"):
                continue
            if current_client is None:
                continue
            for token in line.split(","):
                token = token.strip()
                if token and token.isdigit():
                    clients[current_client].append(int(token))
    return clients


def parse_latest_user_lists(log_path):
    new_user_list = None
    attacker_list = None
    user_pattern = re.compile(r"new_usr_list:\s*(.*)")
    attacker_pattern = re.compile(r"attacker_list:\s*(.*)")

    with open(log_path) as f:
        for line in f:
            user_match = user_pattern.search(line)
            if user_match:
                payload = user_match.group(1).strip()
                new_user_list = [int(item.strip()) for item in payload.split(",") if item.strip()]
            attacker_match = attacker_pattern.search(line)
            if attacker_match:
                payload = attacker_match.group(1).strip()
                attacker_list = [int(item.strip()) for item in payload.split(",") if item.strip()]

    if new_user_list is None:
        raise ValueError(f"Could not parse new_usr_list from {log_path}")
    if attacker_list is None:
        attacker_list = []
    return new_user_list, attacker_list


def load_index_set(path):
    values = set()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                values.add(int(line))
    return values


def build_trigger_tensors(device):
    full_image = torch.zeros((3, 32, 32), dtype=torch.float32)
    full_image.fill_(MASK_VALUE)
    x_bot = BACKDOOR_X_TOP + BACKDOOR_PATTERN.shape[0]
    y_bot = BACKDOOR_Y_TOP + BACKDOOR_PATTERN.shape[1]
    full_image[:, BACKDOOR_X_TOP:x_bot, BACKDOOR_Y_TOP:y_bot] = BACKDOOR_PATTERN

    mask = (full_image != MASK_VALUE).to(torch.float32).to(device)
    pattern = transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)(full_image).to(device)
    return pattern, mask


def apply_trigger(batch_inputs, pattern, mask):
    return (1 - mask) * batch_inputs + mask * pattern


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    for inputs, labels, _ in loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        logits = model(inputs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_samples += labels.size(0)

    return total_loss / max(1, total_samples), 100.0 * total_correct / max(1, total_samples)


def eval_classifier(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for inputs, labels, _ in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            logits = model(inputs)
            loss = criterion(logits, labels)
            total_loss += loss.item() * labels.size(0)
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_samples += labels.size(0)
    return total_loss / max(1, total_samples), 100.0 * total_correct / max(1, total_samples)


def collect_scores(model, loader, device, backdoor_label, pattern, mask):
    rows = []
    model.eval()
    with torch.no_grad():
        for inputs, labels, dataset_indices in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            triggered_inputs = apply_trigger(inputs, pattern, mask)

            clean_logits = model(inputs)
            triggered_logits = model(triggered_inputs)

            clean_probs = torch.softmax(clean_logits, dim=1)
            triggered_probs = torch.softmax(triggered_logits, dim=1)
            clean_target = clean_probs[:, backdoor_label]
            triggered_target = triggered_probs[:, backdoor_label]
            score = triggered_target - clean_target
            clean_pred = clean_logits.argmax(dim=1)
            triggered_pred = triggered_logits.argmax(dim=1)

            for index, label, clean_prob, trig_prob, score_value, clean_pred_value, trig_pred_value in zip(
                dataset_indices.tolist(),
                labels.detach().cpu().tolist(),
                clean_target.detach().cpu().tolist(),
                triggered_target.detach().cpu().tolist(),
                score.detach().cpu().tolist(),
                clean_pred.detach().cpu().tolist(),
                triggered_pred.detach().cpu().tolist(),
            ):
                rows.append(
                    {
                        "dataset_index": int(index),
                        "label": int(label),
                        "clean_target_prob": float(clean_prob),
                        "triggered_target_prob": float(trig_prob),
                        "trigger_delta": float(score_value),
                        "clean_pred": int(clean_pred_value),
                        "triggered_pred": int(trig_pred_value),
                    }
                )
    return rows


def build_score(row, score_mode):
    if score_mode == "all_suspicious_poison":
        return 1.0
    if score_mode == "trigger_delta":
        return float(row["trigger_delta"])
    if score_mode == "clean_minus_trigger":
        return float(row["clean_target_prob"] - row["triggered_target_prob"])
    if score_mode == "triggered_target_prob":
        return float(row["triggered_target_prob"])
    if score_mode == "clean_target_prob":
        return float(row["clean_target_prob"])
    if score_mode == "target_prob_sum":
        return float(row["clean_target_prob"] + row["triggered_target_prob"])
    if score_mode == "abs_delta":
        return float(abs(row["trigger_delta"]))
    raise ValueError(f"Unsupported score mode: {score_mode}")


def percentile_threshold(clean_rows, quantile, score_mode):
    scores = np.array([build_score(row, score_mode) for row in clean_rows], dtype=np.float32)
    return float(np.quantile(scores, quantile))


def evaluate_detector(rows, poison_index_set, threshold, score_mode, threshold_side):
    tp = fp = tn = fn = 0
    predicted_poison_indices = []
    predicted_clean_indices = []
    detailed_rows = []

    for row in rows:
        dataset_index = row["dataset_index"]
        ground_truth_poison = int(dataset_index in poison_index_set)
        score_value = build_score(row, score_mode)
        if threshold_side == "all":
            predicted_poison = 1
        elif threshold_side == "high":
            predicted_poison = int(score_value >= threshold)
        elif threshold_side == "low":
            predicted_poison = int(score_value <= threshold)
        else:
            raise ValueError(f"Unsupported threshold side: {threshold_side}")

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
        updated["score"] = score_value
        updated["ground_truth_poison"] = ground_truth_poison
        updated["predicted_poison"] = predicted_poison
        detailed_rows.append(updated)

    total = tp + fp + tn + fn
    accuracy_ratio = (tp + tn) / max(1, total)
    precision_ratio = tp / max(1, tp + fp)
    recall_ratio = tp / max(1, tp + fn)
    f1_ratio = 0.0 if precision_ratio + recall_ratio == 0 else (
        2 * precision_ratio * recall_ratio / (precision_ratio + recall_ratio)
    )
    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "accuracy": 100.0 * accuracy_ratio,
        "precision": 100.0 * precision_ratio,
        "recall": 100.0 * recall_ratio,
        "f1": 100.0 * f1_ratio,
        "predicted_poison_indices": predicted_poison_indices,
        "predicted_clean_indices": predicted_clean_indices,
        "rows": detailed_rows,
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
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--threshold-quantile", type=float, default=0.50)
    parser.add_argument(
        "--score-mode",
        default="trigger_delta",
        choices=[
            "trigger_delta",
            "clean_minus_trigger",
            "triggered_target_prob",
            "clean_target_prob",
            "target_prob_sum",
            "abs_delta",
            "all_suspicious_poison",
        ],
    )
    parser.add_argument("--threshold-side", default="low", choices=["low", "high", "all"])
    parser.add_argument("--backdoor-label", type=int, default=9)
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_client_path = os.path.join(args.run_dir, "all_client.txt")
    log_path = os.path.join(args.run_dir, "log.txt")
    gt_poison_path = os.path.join(args.audit_dir, "ground_truth_poison_indices.txt")

    client_indices = parse_client_indices(all_client_path)
    good_clients, suspicious_clients = parse_latest_user_lists(log_path)
    poison_index_set = load_index_set(gt_poison_path)

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

    train_loader = DataLoader(IndexedSubset(train_base, train_indices), batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(IndexedSubset(eval_base, val_indices), batch_size=args.batch_size, shuffle=False, num_workers=0)
    suspicious_loader = DataLoader(IndexedSubset(eval_base, suspicious_indices), batch_size=args.batch_size, shuffle=False, num_workers=0)

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
            f"[clean_ref] epoch={epoch} train_loss={train_loss:.4f} train_acc={train_acc:.2f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.2f}"
        )

    pattern, mask = build_trigger_tensors(device)
    clean_rows = collect_scores(model, val_loader, device, args.backdoor_label, pattern, mask)
    suspicious_rows = collect_scores(model, suspicious_loader, device, args.backdoor_label, pattern, mask)
    if args.score_mode == "all_suspicious_poison":
        threshold = None
    else:
        threshold = percentile_threshold(clean_rows, args.threshold_quantile, args.score_mode)

    suspicious_metrics = evaluate_detector(
        suspicious_rows,
        poison_index_set,
        threshold,
        args.score_mode,
        args.threshold_side,
    )
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
        "threshold_side": args.threshold_side,
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

    with open(os.path.join(args.output_dir, "suspicious_scores.csv"), "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "dataset_index",
                "label",
                "clean_target_prob",
                "triggered_target_prob",
                "trigger_delta",
                "score",
                "clean_pred",
                "triggered_pred",
                "ground_truth_poison",
                "predicted_poison",
            ],
        )
        writer.writeheader()
        writer.writerows(suspicious_metrics["rows"])

    write_index_file(os.path.join(args.output_dir, "predicted_poison_indices.txt"), all_poison_filtered)
    write_index_file(os.path.join(args.output_dir, "predicted_clean_indices.txt"), all_clean_kept)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
