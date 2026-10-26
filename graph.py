import re
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from pathlib import Path


def parse_train_log(log_path):
    rows = []
    current_epoch = None
    pending_val = None

    patterns = {
        "loss": r"(?:Train Loss|Val Loss):\s*(-?\d+(?:\.\d+)?)",
        "pos": r"Pos:\s*(-?\d+(?:\.\d+)?)",
        "neg": r"Neg:\s*(-?\d+(?:\.\d+)?)",
        "gap": r"Gap:\s*(-?\d+(?:\.\d+)?)",
        "auc": r"AUC:\s*(-?\d+(?:\.\d+)?)",
        "mrr": r"MRR:\s*(-?\d+(?:\.\d+)?)",
        "map": r"MAP:\s*(-?\d+(?:\.\d+)?)",
        "top1": r"Top1:\s*(-?\d+(?:\.\d+)?)",
        "topk": r"Topk:\s*(-?\d+(?:\.\d+)?)",
        "triplet_loss": r"Triplet Loss:\s*(-?\d+(?:\.\d+)?)",
        "ap_loss": r"AP Loss:\s*(-?\d+(?:\.\d+)?)",
        "local_loss": r"Local Loss:\s*(-?\d+(?:\.\d+)?)",
    }

    def extract_metrics(line, row):
        for key, pattern in patterns.items():
            match = re.search(pattern, line)
            if match is not None:
                row[key] = float(match.group(1))

    with open(log_path, "r") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            if line.startswith("Epoch"): # Train line
                match = re.search(r"Epoch\s+(\d+)", line)
                if match is None:
                    continue

                current_epoch = int(match.group(1))
                row = {"epoch": current_epoch, "split": "train"}

                extract_metrics(line, row)
                rows.append(row)
            
            elif line.startswith("Val Loss"): # Validation first line
                temp = {"epoch": current_epoch, "split": "val"}
                extract_metrics(line, temp)

                if "auc" in temp: # new format
                    rows.append(temp)
                    temp = None


            elif line.startswith("Glob AUC:"):
                if temp is None:
                    continue

                extract_metrics(line, temp)

                rows.append(temp)
                temp = None

    return pd.DataFrame(rows)


def plot_metric_trend(df, y_key, save_dir="plots", y_decimals=4):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    train_df = df[df["split"] == "train"].sort_values("epoch")
    val_df = df[df["split"] == "val"].sort_values("epoch")

    plt.figure(figsize=(6, 6))

    if y_key in train_df.columns:
        plt.plot(
            train_df["epoch"],
            train_df[y_key],
            marker="o",
            markersize=2,
            linestyle="-",
            label="Train",
        )

    if y_key in val_df.columns:
        plt.plot(
            val_df["epoch"],
            val_df[y_key],
            marker="o",
            markersize=2,
            linestyle="-",
            label="Validation",
        )

    plt.xlabel("Epoch")
    plt.ylabel(y_key.upper())
    plt.title(f"{y_key.upper()} Trend")
    plt.legend()
    plt.grid(True, alpha=0.1)

    plt.gca().yaxis.set_major_formatter(FormatStrFormatter(f"%.{y_decimals}f"))

    plt.tight_layout()

    save_path = save_dir / f"{y_key}_trend.png"
    plt.savefig(save_path, dpi=300)
    plt.show()
    plt.close()

    print(f"Saved: {save_path}")


def parse_eval_log(log_path):
    records = []
    current_epoch = None

    with open(log_path, "r") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            # Epoch 5, Epoch 005, Epoch [5], etc.
            if line.lower().startswith("epoch"):
                match = re.search(r"(\d+)", line)

                if match is not None:
                    current_epoch = int(match.group(1))

                continue

            if current_epoch is None:
                continue

            parts = [part.strip() for part in line.split("|")]

            if len(parts) < 2:
                continue

            record = {
                "epoch": current_epoch,
                "dataset": parts[0].split()[0],
            }

            for part in parts[1:]:
                if ":" not in part:
                    continue

                key, value = part.split(":", 1)

                key = key.strip().lower().replace(" ", "_")
                value = value.strip()

                try:
                    record[key] = float(value)
                except ValueError:
                    continue

            records.append(record)

    return records


def plot_eval_metric_by_dataset(records, metric, save_dir="eval_plots"):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    datasets = sorted(set(record["dataset"] for record in records))

    for dataset in datasets:
        dataset_records = [record for record in records if record["dataset"] == dataset and metric in record]
        dataset_records = sorted(dataset_records, key=lambda x: x["epoch"])

        if len(dataset_records) == 0:
            continue

        epochs = [record["epoch"] for record in dataset_records]
        values = [record[metric] for record in dataset_records]

        plt.figure(figsize=(6, 6))
        plt.plot(epochs, values, marker="o", markersize=3)

        plt.xlabel("Epoch")
        plt.ylabel(metric.upper())
        plt.title(f"{dataset} - {metric.upper()}")
        plt.grid(True, alpha=0.2)
        plt.tight_layout()

        save_path = save_dir / f"{dataset}_{metric}.png"
        plt.savefig(save_path, dpi=300)
        plt.close()


def parse_ci_eval_log(log_path, start_epoch=5, epoch_step=5):
    rows = []

    with open(log_path, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    for i, line in enumerate(lines):
        # every 2 lines correspond to one epoch
        epoch = start_epoch + (i // 2) * epoch_step

        parts = [p.strip() for p in line.split("|")]

        dataset = parts[0]

        row = {
            "epoch": epoch,
            "dataset": dataset,
        }

        for part in parts[1:]:
            key, value = part.split(":", 1)

            key = key.strip().lower()
            value = float(value.strip())

            key_map = {
                "pos": "positive_cosine_mean",
                "neg": "negative_cosine_mean",
                "gap": "cosine_gap",
                "pos std": "positive_cosine_std",
                "neg std": "negative_cosine_std",
                "auc": "auc",
                "threshold": "threshold",
                "acc": "accuracy",
                "prec": "precision",
                "recall": "recall",
                "f1": "f1",
                "tp": "tp",
                "fp": "fp",
                "tn": "tn",
                "fn": "fn",
            }

            row[key_map.get(key, key)] = value

        rows.append(row)

    return pd.DataFrame(rows)


def plot_ci_metric_by_dataset(df, metric, save_dir="ci_eval_plots"):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    datasets = sorted(df["dataset"].unique())

    for dataset in datasets:
        dataset_df = df[df["dataset"] == dataset].copy()

        if metric not in dataset_df.columns:
            continue

        dataset_df = dataset_df.sort_values("epoch")

        epochs = dataset_df["epoch"].tolist()
        values = dataset_df[metric].tolist()

        plt.figure(figsize=(6, 6))
        plt.plot(epochs, values, marker="o", markersize=3)

        plt.xlabel("Epoch")
        plt.ylabel(metric.replace("_", " ").title())
        plt.title(f"{dataset} - {metric.replace('_', ' ').title()}")

        plt.xticks(epochs)
        plt.grid(True, alpha=0.2)
        plt.tight_layout()

        save_path = save_dir / f"{dataset}_ci_{metric}.png"
        plt.savefig(save_path, dpi=300)
        plt.close()


# this plots all ci metric at once
def plot_all_ci_metrics_by_dataset(
    df,
    metrics=None,
    save_dir="ci_eval_plots",
):
    if metrics is None:
        metrics = ["auc", "accuracy"]
        # metrics = [
        #     "auc",
        #     "threshold",
        #     "accuracy",
        #     "precision",
        #     "recall",
        #     "f1",
        # ]

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    datasets = sorted(df["dataset"].unique())

    for dataset in datasets:
        dataset_df = (
            df[df["dataset"] == dataset]
            .copy()
            .sort_values("epoch")
        )

        if len(dataset_df) == 0:
            continue

        epochs = dataset_df["epoch"].tolist()

        plt.figure(figsize=(8, 6))

        for metric in metrics:
            if metric not in dataset_df.columns:
                continue

            values = dataset_df[metric].tolist()

            plt.plot(
                epochs,
                values,
                marker="o",
                markersize=10,
                linewidth=5,
                label=metric.upper(),
            )

        plt.xlabel("Epoch")
        plt.ylabel("Score")
        plt.title(f"{dataset} - CI Metrics")

        plt.xticks(epochs)
        plt.ylim(0, 1)

        plt.grid(True, alpha=0.2)
        plt.legend()
        plt.tight_layout()

        save_path = save_dir / f"{dataset}_ci_metrics.png"

        plt.savefig(
            save_path,
            dpi=300,
        )

        plt.close()


def glob_pair_heatmap(save_dir, dset_name, epoch_name, all_src_emb, all_tgt_emb):
    all_src_emb = F.normalize(all_src_emb, dim=-1)
    all_tgt_emb = F.normalize(all_tgt_emb, dim=-1)

    sim_matrix = all_src_emb @ all_tgt_emb.T
    plt.figure(figsize=(10, 8))
    plt.imshow(
        sim_matrix.detach().cpu().numpy(),
        aspect="auto",
        origin="lower",
    )
    plt.colorbar(label="Global Cosine Similarity")
    plt.xlabel("Target Song Index")
    plt.ylabel("Source Song Index")
    plt.title(f"{dset_name} - Global Similarity")
    plt.tight_layout()

    save_path = save_dir / f"{dset_name}_{epoch_name.split(".")[0][-3:]}_glob_cos_sim.png"

    plt.savefig(
        save_path,
        dpi=300,
    )

def local_pair_heatmap(save_dir, dset_name, epoch_name, src_local, tgt_local):
    """
    Compare every selected source local segment with every selected
    target local segment.

    src_local: [N, W, D]
    tgt_local: [M, W, D]

    Returns:
        score_matrix: [N, M]
    """

    # Normalize just in case
    src_local = F.normalize(src_local, p=2, dim=-1)
    tgt_local = F.normalize(tgt_local, p=2, dim=-1)

    print("src local:", src_local.shape)
    print("tgt local:", tgt_local.shape)

    # [N, W, D] x [M, W, D]
    # -> [N, M, W, W]
    pair_sim = torch.einsum(
        "nkd,mld->nmkl",
        src_local,
        tgt_local
    )

    # Take corresponding positions within each selected window:
    #
    # src[0] <-> tgt[0]
    # src[1] <-> tgt[1]
    # ...
    #
    # [N, M, W]
    diagonal_sim = pair_sim.diagonal(
        dim1=-2,
        dim2=-1
    )

    # Mean similarity of the W aligned local embeddings
    # [N, M]
    score_matrix = diagonal_sim.mean(dim=-1)

    print("local score matrix:", score_matrix.shape)
    print(
        "score min/mean/max:",
        score_matrix.min().item(),
        score_matrix.mean().item(),
        score_matrix.max().item()
    )

    # Save heatmap
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    save_path = (save_dir / f"{dset_name}_{epoch_name.split(".")[0][-3:]}_local_similarity.png")

    plt.figure(figsize=(10, 8))

    plt.imshow(
        score_matrix.detach().cpu().numpy(),
        aspect="auto",
        origin="lower"
    )

    plt.colorbar(label="Local Cosine Similarity")
    plt.xlabel("Target Song Index")
    plt.ylabel("Source Song Index")
    plt.title(f"{dset_name} - Local Similarity")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()