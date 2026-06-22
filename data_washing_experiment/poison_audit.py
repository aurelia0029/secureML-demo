import csv
import json
import os
from collections import defaultdict
import random

import torch
from torch.utils.data import Dataset


class IndexedDataset(Dataset):
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, index):
        sample = self.base_dataset[index]
        if isinstance(sample, tuple):
            return (*sample, index)
        raise TypeError(f"Unsupported dataset sample type: {type(sample)!r}")


def ensure_indexed_dataset(dataset):
    if dataset is None or isinstance(dataset, IndexedDataset):
        return dataset
    return IndexedDataset(dataset)


def unwrap_indexed_sample(sample):
    if not isinstance(sample, tuple):
        raise TypeError(f"Unsupported dataset sample type: {type(sample)!r}")
    if len(sample) >= 3:
        return sample[0], sample[1], int(sample[-1])
    if len(sample) == 2:
        return sample[0], sample[1], None
    raise ValueError(f"Unsupported dataset sample length: {len(sample)}")


def ensure_poison_tracking_state(task):
    if not hasattr(task, "poisoned_train_samples"):
        task.poisoned_train_samples = set()
    if not hasattr(task, "poisoned_train_samples_by_client"):
        task.poisoned_train_samples_by_client = defaultdict(set)
    if not hasattr(task, "fixed_poison_dataset_indices"):
        task.fixed_poison_dataset_indices = set()
    if not hasattr(task, "fixed_poison_dataset_indices_by_client"):
        task.fixed_poison_dataset_indices_by_client = {}


def ensure_fixed_poison_selection(hlpr, client_id):
    ensure_poison_tracking_state(hlpr.task)
    if not getattr(hlpr.params, "fixed_poison_dataset", False):
        return set()
    if client_id is None or not hasattr(hlpr.task, "getClientDataIndex"):
        return set()

    client_id = int(client_id)
    if client_id in hlpr.task.fixed_poison_dataset_indices_by_client:
        return hlpr.task.fixed_poison_dataset_indices_by_client[client_id]

    client_indices = list(hlpr.task.getClientDataIndex(client_id))
    poison_count = round(len(client_indices) * hlpr.params.poisoning_proportion)
    poison_count = max(0, min(len(client_indices), poison_count))

    seed = getattr(hlpr.params, "fixed_poison_seed", None)
    if seed is None:
        seed = getattr(hlpr.params, "random_seed", 0) or 0
    rng = random.Random(int(seed) + client_id * 9973)
    selected = set(rng.sample(client_indices, poison_count)) if poison_count > 0 else set()

    hlpr.task.fixed_poison_dataset_indices_by_client[client_id] = selected
    hlpr.task.fixed_poison_dataset_indices.update(selected)
    hlpr.task.poisoned_train_samples.update(selected)
    hlpr.task.poisoned_train_samples_by_client[client_id].update(selected)
    return selected


def record_poisoned_batch(hlpr, batch, client_id=None):
    ensure_poison_tracking_state(hlpr.task)
    if batch.aux is None:
        return

    if getattr(hlpr.params, "fixed_poison_dataset", False):
        fixed_indices = ensure_fixed_poison_selection(hlpr, client_id)
        if not fixed_indices:
            return
        poisoned_indices = [
            int(idx)
            for idx in batch.aux.detach().cpu().tolist()
            if int(idx) in fixed_indices
        ]
        if client_id is not None:
            hlpr.task.poisoned_train_samples_by_client[int(client_id)].update(poisoned_indices)
        return

    poison_count = round(batch.batch_size * hlpr.params.poisoning_proportion)
    if poison_count <= 0:
        return

    poisoned_indices = batch.aux[:poison_count].detach().cpu().tolist()
    poisoned_indices = [int(idx) for idx in poisoned_indices]
    hlpr.task.poisoned_train_samples.update(poisoned_indices)

    if client_id is not None:
        hlpr.task.poisoned_train_samples_by_client[int(client_id)].update(poisoned_indices)


def apply_trigger_to_input(hlpr, input_tensor):
    pattern, mask = hlpr.attack.synthesizer.get_pattern()
    device = input_tensor.device
    pattern = pattern.to(device)
    mask = mask.to(device)
    return (1 - mask) * input_tensor + mask * pattern


def _predict_poison_status(hlpr, logits, original_labels):
    probs = torch.softmax(logits, dim=1)
    pred_classes = torch.argmax(logits, dim=1)
    poison_scores = probs[:, hlpr.params.backdoor_label]

    predicted_status = []
    for pred_class, original_label, poison_score in zip(
        pred_classes.detach().cpu().tolist(),
        original_labels.detach().cpu().tolist(),
        poison_scores.detach().cpu().tolist(),
    ):
        is_poison = int(
            pred_class == hlpr.params.backdoor_label
            and (original_label != hlpr.params.backdoor_label or poison_score >= 0.90)
        )
        predicted_status.append(is_poison)

    return pred_classes.detach().cpu().tolist(), poison_scores.detach().cpu().tolist(), predicted_status


def run_final_poison_audit(hlpr):
    if not getattr(hlpr.params, "data_washing_audit", False):
        return None
    if not getattr(hlpr.params, "fl", False):
        return None
    if not hasattr(hlpr.task, "getClientDataIndex"):
        return None

    ensure_poison_tracking_state(hlpr.task)

    output_dir = getattr(hlpr.params, "data_washing_output_dir", None) or os.path.join(
        hlpr.params.folder_path, "data_washing_audit"
    )
    os.makedirs(output_dir, exist_ok=True)

    model = hlpr.task.model
    model.eval()

    batch_size = int(getattr(hlpr.params, "data_washing_batch_size", 128) or 128)
    poisoned_indices = set(int(idx) for idx in getattr(hlpr.task, "poisoned_train_samples", set()))

    rows = []
    client_summary = []
    predicted_clean_indices = []
    predicted_poison_indices = []
    ground_truth_clean_indices = []
    ground_truth_poison_indices = []

    total_correct = 0
    total_count = 0

    for client_id in range(int(hlpr.params.fl_total_participants)):
        client_indices = list(hlpr.task.getClientDataIndex(client_id))
        client_correct = 0
        client_total = 0
        client_poison_gt = 0
        client_poison_pred = 0

        for start in range(0, len(client_indices), batch_size):
            chunk_indices = client_indices[start:start + batch_size]
            inputs = []
            original_labels = []
            gt_statuses = []

            for dataset_index in chunk_indices:
                sample = hlpr.task.train_dataset[int(dataset_index)]
                input_tensor, label, _ = unwrap_indexed_sample(sample)
                input_tensor = input_tensor.clone()
                label = int(label)
                gt_status = int(dataset_index in poisoned_indices)
                if gt_status:
                    input_tensor = apply_trigger_to_input(hlpr, input_tensor)
                    client_poison_gt += 1
                    ground_truth_poison_indices.append(int(dataset_index))
                else:
                    ground_truth_clean_indices.append(int(dataset_index))

                inputs.append(input_tensor)
                original_labels.append(label)
                gt_statuses.append(gt_status)

            input_batch = torch.stack(inputs, dim=0).to(hlpr.params.device)
            original_label_tensor = torch.tensor(original_labels, dtype=torch.long, device=hlpr.params.device)

            with torch.no_grad():
                if model.__class__.__name__ == "PPNet":
                    logits, _ = model(input_batch)
                elif model.__class__.__name__ == "VGG":
                    logits, _ = model(input_batch)
                else:
                    logits = model(input_batch)

            pred_classes, poison_scores, predicted_statuses = _predict_poison_status(
                hlpr, logits, original_label_tensor
            )

            for dataset_index, original_label, gt_status, pred_class, poison_score, predicted_status in zip(
                chunk_indices,
                original_labels,
                gt_statuses,
                pred_classes,
                poison_scores,
                predicted_statuses,
            ):
                is_correct = int(gt_status == predicted_status)
                total_correct += is_correct
                total_count += 1
                client_correct += is_correct
                client_total += 1
                client_poison_pred += predicted_status

                if predicted_status:
                    predicted_poison_indices.append(int(dataset_index))
                else:
                    predicted_clean_indices.append(int(dataset_index))

                rows.append({
                    "client_id": int(client_id),
                    "dataset_index": int(dataset_index),
                    "original_label": int(original_label),
                    "ground_truth_status": "poison" if gt_status else "clean",
                    "predicted_status": "poison" if predicted_status else "clean",
                    "predicted_class": int(pred_class),
                    "backdoor_score": float(poison_score),
                    "is_correct": int(is_correct),
                })

        client_accuracy = 100.0 * client_correct / max(1, client_total)
        client_summary.append({
            "client_id": int(client_id),
            "samples": int(client_total),
            "ground_truth_poison": int(client_poison_gt),
            "predicted_poison": int(client_poison_pred),
            "accuracy": client_accuracy,
        })

    overall_accuracy = 100.0 * total_correct / max(1, total_count)
    summary = {
        "total_samples": int(total_count),
        "ground_truth_poison_samples": int(len(ground_truth_poison_indices)),
        "ground_truth_clean_samples": int(len(ground_truth_clean_indices)),
        "predicted_poison_samples": int(len(predicted_poison_indices)),
        "predicted_clean_samples": int(len(predicted_clean_indices)),
        "classification_accuracy": overall_accuracy,
        "backdoor_label": int(hlpr.params.backdoor_label),
        "clients": client_summary,
    }

    with open(os.path.join(output_dir, "sample_classification.csv"), "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "client_id",
                "dataset_index",
                "original_label",
                "ground_truth_status",
                "predicted_status",
                "predicted_class",
                "backdoor_score",
                "is_correct",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    with open(os.path.join(output_dir, "client_summary.csv"), "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["client_id", "samples", "ground_truth_poison", "predicted_poison", "accuracy"],
        )
        writer.writeheader()
        writer.writerows(client_summary)

    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    for filename, values in [
        ("ground_truth_clean_indices.txt", ground_truth_clean_indices),
        ("ground_truth_poison_indices.txt", ground_truth_poison_indices),
        ("predicted_clean_indices.txt", predicted_clean_indices),
        ("predicted_poison_indices.txt", predicted_poison_indices),
    ]:
        with open(os.path.join(output_dir, filename), "w") as f:
            for value in values:
                f.write(f"{int(value)}\n")

    return summary
