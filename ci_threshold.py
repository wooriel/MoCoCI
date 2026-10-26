import torch
import torch.nn.functional as F
from sklearn.metrics import roc_curve, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from models.losses import best_diagonal_window
from graph import glob_pair_heatmap, local_pair_heatmap


def calculate_ci_metric(model, dataloader, dset_name, device, save_path, epoch_name):
    all_cosine_sim = []
    all_local_sim = []
    all_labels = []

    all_src_emb = []
    all_tgt_emb = []

    all_loc_src_emb = []
    all_loc_tgt_emb = []

    alpha = 0.5

    for batch in dataloader:
        src = batch["orig_inp"].to(device)
        tgt = batch["cover_inp"].to(device)
        labels = batch["decision"].to(device)

        pair_label = batch["decision"].to(device=device, dtype=torch.float32)
        pos_mask = pair_label == 1

        outputs = model(src, tgt)
        # for visualization
        if pos_mask.any():
            src_emb = outputs["src_feature"][pos_mask]
            tgt_emb = outputs["tgt_feature"][pos_mask]
            all_src_emb.append(src_emb)
            all_tgt_emb.append(tgt_emb)
            loc_src_emb = outputs["src_local"][pos_mask]
            loc_tgt_emb = outputs["tgt_local"][pos_mask]

            pos_sim = outputs["pair_similarity"][pos_mask]
            window = 8

            # print(loc_src_emb.shape)
            # print(loc_tgt_emb.shape)
            # print(pos_sim.shape)
            pos_local_score, src_start, tgt_start = best_diagonal_window(pos_sim, window=window, return_indices=True)
            
            B = loc_src_emb.shape[0]

            offset = torch.arange(window, device=loc_src_emb.device)   # [8]

            src_idx = src_start[:, None] + offset[None, :]   # [B, 8]
            tgt_idx = tgt_start[:, None] + offset[None, :]   # [B, 8]

            batch_idx = torch.arange(B, device=loc_src_emb.device)[:, None]

            # Crop the corresponding embeddings
            src_partial = loc_src_emb[batch_idx, src_idx, :]   # [B, 8, D]
            tgt_partial = loc_tgt_emb[batch_idx, tgt_idx, :]   # [B, 8, D]
            all_loc_src_emb.append(src_partial)
            all_loc_tgt_emb.append(tgt_partial)


        # global feature
        src_feature = outputs["src_feature"]
        tgt_feature = outputs["tgt_feature"]

        cosine_sim = F.cosine_similarity(
            src_feature,
            tgt_feature,
            dim=-1,
        )

        all_cosine_sim.append(cosine_sim.detach().cpu())
        all_labels.append(labels.detach().cpu())

        # local feature
        pair_sim = outputs["pair_similarity"]
        local_score, _, _ = best_diagonal_window(
            pair_sim,
            window=8,
            return_indices=False,
        ) # ideally [B]

        all_local_sim.append(local_score.detach().cpu())


    all_cosine_sim = torch.cat(all_cosine_sim, dim=0)
    all_local_sim = torch.cat(all_local_sim, dim=0)
    all_comb_sim = alpha * all_cosine_sim + (1-alpha) * all_local_sim
    all_labels = torch.cat(all_labels, dim=0)

    pos_mask = all_labels == 1
    neg_mask = all_labels == 0

    pos_sim = all_cosine_sim[pos_mask]
    neg_sim = all_cosine_sim[neg_mask]

    loc_pos_sim = all_local_sim[pos_mask]
    loc_neg_sim = all_local_sim[neg_mask]

    pos_mean = pos_sim.mean().item() if pos_sim.numel() > 0 else float("nan")
    neg_mean = neg_sim.mean().item() if neg_sim.numel() > 0 else float("nan")

    pos_std = pos_sim.std().item() if pos_sim.numel() > 1 else float("nan")
    neg_std = neg_sim.std().item() if neg_sim.numel() > 1 else float("nan")

    loc_pos_mean = loc_pos_sim.mean().item() if loc_pos_sim.numel() > 0 else float("nan")
    loc_neg_mean = loc_neg_sim.mean().item() if loc_neg_sim.numel() > 0 else float("nan")

    loc_pos_std = loc_pos_sim.std().item() if loc_pos_sim.numel() > 1 else float("nan")
    loc_neg_std = loc_neg_sim.std().item() if loc_neg_sim.numel() > 1 else float("nan")

    if pos_mask.any() and neg_mask.any():
        glob_auc = roc_auc_score(
            all_labels.numpy(),
            all_cosine_sim.numpy(),
        )
        loc_auc = roc_auc_score(
            all_labels.numpy(),
            all_local_sim.numpy(),
        )
        comb_auc = roc_auc_score(
            all_labels.numpy(),
            all_comb_sim.numpy(),
        )
    else:
        glob_auc = float("nan")
        loc_auc = float("nan")
        comb_auc = float("nan")
    

    # scores = torch.cat(all_cosine_sim).numpy()
    # labels = torch.cat(all_labels).numpy()
    all_src_emb = torch.cat(all_src_emb, dim=0)
    all_tgt_emb = torch.cat(all_tgt_emb, dim=0)
    all_loc_src_emb = torch.cat(all_loc_src_emb, dim=0)
    all_loc_tgt_emb = torch.cat(all_loc_tgt_emb, dim=0)

    glob_pair_heatmap(save_path, dset_name, epoch_name, all_src_emb, all_tgt_emb)
    local_pair_heatmap(save_path, dset_name, epoch_name, all_loc_src_emb, all_loc_tgt_emb)

    fpr, tpr, thresholds = roc_curve(all_labels, all_cosine_sim)
    loc_fpr, loc_tpr, loc_thresholds = roc_curve(all_labels, all_local_sim)
    comb_fpr, comb_tpr, comb_thresholds = roc_curve(all_labels, all_comb_sim)

    j = tpr - fpr
    best_idx = j.argmax()

    loc_j = loc_tpr - loc_fpr
    loc_best_idx = loc_j.argmax()

    comb_j = comb_tpr - comb_fpr
    comb_best_idx = comb_j.argmax()

    best_threshold = thresholds[best_idx]
    loc_best_threshold = loc_thresholds[loc_best_idx]
    comb_best_threshold = comb_thresholds[comb_best_idx]

    # print("Best threshold:", best_threshold)
    # print("TPR:", tpr[best_idx])
    # print("FPR:", fpr[best_idx])

    pred = (all_cosine_sim >= best_threshold).cpu().numpy().astype(int)
    loc_pred = (all_local_sim >= loc_best_threshold).cpu().numpy().astype(int)
    comb_pred = (all_comb_sim >= comb_best_threshold).cpu().numpy().astype(int)

    accuracy = accuracy_score(all_labels, pred)
    precision = precision_score(all_labels, pred)
    recall = recall_score(all_labels, pred)
    f1 = f1_score(all_labels, pred)
    tn, fp, fn, tp = confusion_matrix(all_labels, pred).ravel()

    loc_accuracy = accuracy_score(all_labels, loc_pred)
    loc_precision = precision_score(all_labels, loc_pred)
    loc_recall = recall_score(all_labels, loc_pred)
    loc_f1 = f1_score(all_labels, loc_pred)
    loc_tn, loc_fp, loc_fn, loc_tp = confusion_matrix(all_labels, loc_pred).ravel()

    comb_accuracy = accuracy_score(all_labels, comb_pred)
    comb_precision = precision_score(all_labels, comb_pred)
    comb_recall = recall_score(all_labels, comb_pred)
    comb_f1 = f1_score(all_labels, comb_pred)
    comb_tn, comb_fp, comb_fn, comb_tp = confusion_matrix(all_labels, comb_pred).ravel()

    return {
        "positive_cosine_mean": pos_mean,
        "negative_cosine_mean": neg_mean,
        "loc_positive_cosine_mean": loc_pos_mean,
        "loc_negative_cosine_mean": loc_neg_mean,

        "positive_cosine_std": pos_std,
        "negative_cosine_std": neg_std,
        "loc_positive_cosine_std": loc_pos_std,
        "loc_negative_cosine_std": loc_neg_std,

        "cosine_gap": pos_mean - neg_mean,
        "loc_cosine_gap": loc_pos_mean - loc_neg_mean,

        "glob_auc": glob_auc,
        "loc_auc": loc_auc,
        "comb_auc": comb_auc,

        "scores": all_cosine_sim,
        "loc_scores": all_local_sim,
        "labels": all_labels,

        "threshold": best_threshold,
        "loc_threshold": loc_best_threshold,
        "comb_threshold": comb_best_threshold,

        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,

        "loc_accuracy": loc_accuracy,
        "loc_precision": loc_precision,
        "loc_recall": loc_recall,
        "loc_f1": loc_f1,
        "loc_tp": loc_tp,
        "loc_fp": loc_fp,
        "loc_tn": loc_tn,
        "loc_fn": loc_fn,

        "comb_accuracy": comb_accuracy,
        "comb_precision": comb_precision,
        "comb_recall": comb_recall,
        "comb_f1": comb_f1,
        "comb_tp": comb_tp,
        "comb_fp": comb_fp,
        "comb_tn": comb_tn,
        "comb_fn": comb_fn,
    }


def calculate_ci_moco_metric(model, dataloader, dset_name, device, save_path, epoch_name):
    all_cosine_sim = []
    all_local_sim = []
    all_labels = []

    all_src_emb = []
    all_tgt_emb = []

    all_loc_src_emb = []
    all_loc_tgt_emb = []

    alpha = 0.5

    for batch in dataloader:
        src = batch["orig_inp"].to(device)
        tgt = batch["cover_inp"].to(device)
        labels = batch["decision"].to(device=device, dtype=torch.long)

        pos_mask = labels == 1

        output = model(src, tgt)
        if pos_mask.any():
            src_emb = output["x1_query_pred"][pos_mask]
            tgt_emb = output["x2_key_feature"][pos_mask]
            all_src_emb.append(src_emb)
            all_tgt_emb.append(tgt_emb)
            loc_src_emb = output["x1_local_feature"][pos_mask]
            loc_tgt_emb = output["x2_local_feature"][pos_mask]

            pos_sim = output["pair_sim"][pos_mask]
            window = 8

            pos_local_score, src_start, tgt_start = best_diagonal_window(pos_sim, window=window, return_indices=True)
            
            B = loc_src_emb.shape[0]

            offset = torch.arange(window, device=loc_src_emb.device)   # [8]

            src_idx = src_start[:, None] + offset[None, :]   # [B, 8]
            tgt_idx = tgt_start[:, None] + offset[None, :]   # [B, 8]

            batch_idx = torch.arange(B, device=loc_src_emb.device)[:, None]

            # Crop the corresponding embeddings
            src_partial = loc_src_emb[batch_idx, src_idx, :]   # [B, 8, D]
            tgt_partial = loc_tgt_emb[batch_idx, tgt_idx, :]   # [B, 8, D]
            all_loc_src_emb.append(src_partial)
            all_loc_tgt_emb.append(tgt_partial)

        # MoCo query/key features
        q1 = F.normalize(output["x1_query_pred"], dim=-1)
        q2 = F.normalize(output["x2_query_pred"], dim=-1)

        k1 = F.normalize(output["x1_key_feature"], dim=-1)
        k2 = F.normalize(output["x2_key_feature"], dim=-1)

        # src -> tgt
        sim_12 = F.cosine_similarity(q1, k2, dim=-1)

        # tgt -> src
        sim_21 = F.cosine_similarity(q2, k1, dim=-1)

        # symmetric pair score
        cosine_sim = 0.5 * (sim_12 + sim_21)

        # local feature
        pair_sim = output["pair_sim"]
        local_score, _, _ = best_diagonal_window(
            pair_sim,
            window=8,
            return_indices=False,
        ) # ideally [B]

        all_cosine_sim.append(cosine_sim.detach().cpu())
        all_local_sim.append(local_score.detach().cpu())
        all_labels.append(labels.detach().cpu())

    all_cosine_sim = torch.cat(all_cosine_sim, dim=0)
    all_local_sim = torch.cat(all_local_sim, dim=0)
    all_comb_sim = alpha * all_cosine_sim + (1-alpha) * all_local_sim
    all_labels = torch.cat(all_labels, dim=0)

    # --------------------------------------------------
    # Positive / negative cosine statistics
    # --------------------------------------------------

    pos_mask = all_labels == 1
    neg_mask = all_labels == 0

    pos_sim = all_cosine_sim[pos_mask]
    neg_sim = all_cosine_sim[neg_mask]

    loc_pos_sim = all_local_sim[pos_mask]
    loc_neg_sim = all_local_sim[neg_mask]

    pos_mean = pos_sim.mean().item() if pos_sim.numel() > 0 else float("nan")
    neg_mean = neg_sim.mean().item() if neg_sim.numel() > 0 else float("nan")

    loc_pos_mean = loc_pos_sim.mean().item() if loc_pos_sim.numel() > 0 else float("nan")
    loc_neg_mean = loc_neg_sim.mean().item() if loc_neg_sim.numel() > 0 else float("nan")

    pos_std = pos_sim.std().item() if pos_sim.numel() > 1 else float("nan")
    neg_std = neg_sim.std().item() if neg_sim.numel() > 1 else float("nan")

    loc_pos_std = loc_pos_sim.std().item() if loc_pos_sim.numel() > 1 else float("nan")
    loc_neg_std = loc_neg_sim.std().item() if loc_neg_sim.numel() > 1 else float("nan")

    # Convert to NumPy once for sklearn
    all_cosine_sim = all_cosine_sim.numpy()
    all_local_sim = all_local_sim.numpy()
    all_comb_sim = all_comb_sim.numpy()
    all_labels = all_labels.numpy()

    # ROC-AUC
    glob_auc = roc_auc_score(
        all_labels,
        all_cosine_sim,
    )
    loc_auc = roc_auc_score(
        all_labels,
        all_local_sim,
    )
    comb_auc = roc_auc_score(
        all_labels,
        all_comb_sim,
    )

    all_src_emb = torch.cat(all_src_emb, dim=0)
    all_tgt_emb = torch.cat(all_tgt_emb, dim=0)
    all_loc_src_emb = torch.cat(all_loc_src_emb, dim=0)
    all_loc_tgt_emb = torch.cat(all_loc_tgt_emb, dim=0)

    glob_pair_heatmap(save_path, dset_name, epoch_name, all_src_emb, all_tgt_emb)
    local_pair_heatmap(save_path, dset_name, epoch_name, all_loc_src_emb, all_loc_tgt_emb)

    fpr, tpr, thresholds = roc_curve(all_labels, all_cosine_sim)
    loc_fpr, loc_tpr, loc_thresholds = roc_curve(all_labels, all_local_sim)
    comb_fpr, comb_tpr, comb_thresholds = roc_curve(all_labels, all_comb_sim)

    j = tpr - fpr
    best_idx = j.argmax()

    loc_j = loc_tpr - loc_fpr
    loc_best_idx = loc_j.argmax()

    comb_j = comb_tpr - comb_fpr
    comb_best_idx = comb_j.argmax()

    best_threshold = thresholds[best_idx]
    loc_best_threshold = loc_thresholds[loc_best_idx]
    comb_best_threshold = comb_thresholds[comb_best_idx]

    # Binary prediction
    pred = (all_cosine_sim >= best_threshold).astype(int)
    loc_pred = (all_local_sim >= loc_best_threshold).astype(int)
    comb_pred = (all_comb_sim >= comb_best_threshold).astype(int)

    accuracy = accuracy_score(all_labels, pred)
    precision = precision_score(all_labels, pred)
    recall = recall_score(all_labels, pred)
    f1 = f1_score(all_labels, pred)
    tn, fp, fn, tp = confusion_matrix(all_labels, pred).ravel()

    loc_accuracy = accuracy_score(all_labels, loc_pred)
    loc_precision = precision_score(all_labels, loc_pred)
    loc_recall = recall_score(all_labels, loc_pred)
    loc_f1 = f1_score(all_labels, loc_pred)
    loc_tn, loc_fp, loc_fn, loc_tp = confusion_matrix(all_labels, loc_pred).ravel()

    comb_accuracy = accuracy_score(all_labels, comb_pred)
    comb_precision = precision_score(all_labels, comb_pred)
    comb_recall = recall_score(all_labels, comb_pred)
    comb_f1 = f1_score(all_labels, comb_pred)
    comb_tn, comb_fp, comb_fn, comb_tp = confusion_matrix(all_labels, comb_pred).ravel()

    return {
        "positive_cosine_mean": pos_mean,
        "negative_cosine_mean": neg_mean,
        "loc_positive_cosine_mean": loc_pos_mean,
        "loc_negative_cosine_mean": loc_neg_mean,

        "positive_cosine_std": pos_std,
        "negative_cosine_std": neg_std,
        "loc_positive_cosine_std": loc_pos_std,
        "loc_negative_cosine_std": loc_neg_std,

        "cosine_gap": pos_mean - neg_mean,
        "loc_cosine_gap": loc_pos_mean - loc_neg_mean,

        "glob_auc": glob_auc,
        "loc_auc": loc_auc,
        "comb_auc": comb_auc,

        "scores": all_cosine_sim,
        "loc_scores": all_local_sim,
        "labels": all_labels,

        "threshold": best_threshold,
        "loc_threshold": loc_best_threshold,
        "comb_threshold": comb_best_threshold,

        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,

        "loc_accuracy": loc_accuracy,
        "loc_precision": loc_precision,
        "loc_recall": loc_recall,
        "loc_f1": loc_f1,
        "loc_tp": loc_tp,
        "loc_fp": loc_fp,
        "loc_tn": loc_tn,
        "loc_fn": loc_fn,

        "comb_accuracy": comb_accuracy,
        "comb_precision": comb_precision,
        "comb_recall": comb_recall,
        "comb_f1": comb_f1,
        "comb_tp": comb_tp,
        "comb_fp": comb_fp,
        "comb_tn": comb_tn,
        "comb_fn": comb_fn,
    }


# def log_ci_eval(log_path, dataset_name, metrics):
#     log_line = (
#         "{} | "
#         "Pos: {:.4f} | Neg: {:.4f} | Gap: {:.4f} | "
#         "Pos Std: {:.4f} | Neg Std: {:.4f} | AUC: {:.4f} | "
#         "Threshold: {:.4f} | Acc: {:.4f} | Prec: {:.4f} | Recall: {:.4f} | "
#         "F1: {:.4f} | TP: {:.4f} | FP: {:.4f} | TN: {:.4f} | FN: {:.4f}"
#     ).format(
#         dataset_name,
#         metrics["positive_cosine_mean"],
#         metrics["negative_cosine_mean"],
#         metrics["cosine_gap"],
#         metrics["positive_cosine_std"],
#         metrics["negative_cosine_std"],
#         metrics["roc_auc"],
#         metrics['threshold'],
#         metrics['accuracy'],
#         metrics['precision'],
#         metrics['recall'],
#         metrics['f1'],
#         metrics['tp'],
#         metrics['fp'],
#         metrics['tn'],
#         metrics['fn'],
#     )

#     print(log_line)

#     with open(log_path, "a") as f:
#         f.write(log_line + "\n")

def log_ci_eval(log_path, dataset_name, metrics):
    log_line = (
        "{} | "
        "Glob AUC: {:.4f} | Loc AUC: {:.4f} | Comb AUC: {:.4f} | "
        "Threshold: {:.4f} | Loc Threshold: {:.4f} | Comb Threshold: {:.4f} | "
        "Acc: {:.4f} | Loc Acc: {:.4f} | Comb Acc: {:.4f} | "
        "Prec: {:.4f} | Recall: {:.4f} | F1: {:.4f} | "
        "Loc Prec: {:.4f} | Loc Recall: {:.4f} | Loc F1: {:.4f} | "
        "Comb Prec: {:.4f} | Comb Recall: {:.4f} | Comb F1: {:.4f}"
    ).format(
        dataset_name,
        metrics['glob_auc'],
        metrics['loc_auc'],
        metrics['comb_auc'],
        metrics['threshold'],
        metrics['loc_threshold'],
        metrics['comb_threshold'],
        metrics['accuracy'],
        metrics['loc_accuracy'],
        metrics['comb_accuracy'],
        metrics['precision'],
        metrics['recall'],
        metrics['f1'],
        metrics['loc_precision'],
        metrics['loc_recall'],
        metrics['loc_f1'],
        metrics['comb_precision'],
        metrics['comb_recall'],
        metrics['comb_f1'],
    )

    print(log_line)

    with open(log_path, "a") as f:
        f.write(log_line + "\n")