from tqdm import tqdm
from models.f_losses import moco_contrastive_loss, supervised_moco_loss, smooth_ap_loss, triplet_loss, retrieval_loss, retrieval_metrics_val, best_diagonal_window
from score_metric import get_moco_mrr
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_curve, roc_auc_score


def train_vit_moco_one_epoch(model, dataloader, optimizer, device):
    model.train()

    total_loss = 0.0
    total_samples = 0

    lambda_moco = 0.5
    lambda_local = 0.3
    lambda_ap = 1.0

    total_loc = 0.0
    total_ap = 0.0

    progress_bar = tqdm(dataloader, desc="Training", leave=False)

    for batch in progress_bar:
        x1_work_id = batch["src_work_id"].to(device)
        x2_work_id = batch["pos_work_id"].to(device)

        x1_perf_id = batch["src_label"].to(device)
        x2_perf_id = batch["pos_label"].to(device)

        x1 = batch["orig_inp"].to(device)
        x2 = batch["pos_inp"].to(device)
        neg_x2 = batch["neg_inp"].to(device)

        optimizer.zero_grad()

        output = model(x1, x2)
        outputn = model(x1, neg_x2)

        # 1. flatten to have same pair on the batch
        query_pred = torch.cat([output["x1_query_pred"], output["x2_query_pred"]], dim=0) # [2B, D]
        key_feature = torch.cat([output["x1_key_feature"], output["x2_key_feature"]], dim=0) # [2B, D]

        work_ids = torch.cat([x1_work_id, x2_work_id], dim=0) # [2B]
        perf_ids = torch.cat([x1_perf_id, x2_perf_id], dim=0) # [2B]

        # 2. Identify if there are same work within the batch
        same_work = (work_ids[:, None] == work_ids[None, :]) # [2B, 2B]
        same_perf = (perf_ids[:, None] == perf_ids[None, :]) & same_work

        positive_mask = same_work & ~same_perf # positive: boolean id [2B]
        negative_mask = ~same_work # negative
        candidate_mask = ~same_perf # remove exactly same song

        query_pred = F.normalize(query_pred, dim=-1)
        key_feature = F.normalize(key_feature, dim=-1)

        sim_matrix = query_pred @ key_feature.T # [2B, 2B]

        pos = sim_matrix[positive_mask]
        neg = sim_matrix[negative_mask]

        # smooth AP loss
        embeddings = torch.cat([output["x1_glob_feature"], output["x2_glob_feature"]], dim=0) # [2B, D]
        work_ids = torch.cat([x1_work_id, x2_work_id], dim=0)

        ap_loss = smooth_ap_loss(embeddings, work_ids, temperature=0.01)

        # Top 1
        top1_matrix = sim_matrix.masked_fill(~candidate_mask, float("-inf"))

        pred = top1_matrix.argmax(dim=1) # [2B]
        row_idx = torch.arange(sim_matrix.size(0), device=sim_matrix.device)
        moco_top1 = positive_mask[row_idx, pred].float().mean()

        moco_loss = supervised_moco_loss(
            q=query_pred,
            k=key_feature,
            work_ids=work_ids,
            perf_ids=perf_ids,
            temperature=0.2,
        )

        pos_sim = output["pair_sim"]
        neg_sim = outputn["pair_sim"]
        # neg_out = model.encoder_q.encode_one(neg_x2)
        # neg_local = neg_out["local_feature"]
        
        # _, _, neg_sim = model.encoder_q.pair_projection(
        #     output["x1_local_feature"],
        #     neg_local
        # )
        pos_local_score, _, _ = best_diagonal_window(pos_sim, window=8)
        neg_local_score, _, _ = best_diagonal_window(neg_sim, window=8)
        local_loss = F.relu(neg_local_score - pos_local_score + 0.2).mean()

        loss = lambda_moco * moco_loss + lambda_local * local_loss + lambda_ap * ap_loss

        loss.backward()
        optimizer.step()

        batch_size = work_ids.size(0) # [2B]
        total_loss += loss.item() * batch_size
        total_samples += batch_size

        total_loc += local_loss.item() * batch_size # local loss
        total_ap += ap_loss.item() * batch_size # ap_loss

        # retrieval metrics
        retrieval_sim = sim_matrix.masked_fill(~candidate_mask, float("-inf"))

        # Calculate AUC
        valid_pair_mask = candidate_mask

        auc_scores = sim_matrix[valid_pair_mask]
        auc_labels = positive_mask[valid_pair_mask].long()

        mrr_score, map_score = get_moco_mrr(retrieval_sim, positive_mask)

        auc = roc_auc_score(auc_labels.detach().cpu().numpy(), auc_scores.detach().cpu().numpy(),)

    return {
        "loss": total_loss / total_samples,
        "positive_mean": pos.mean().item(),
        "negative_mean": neg.mean().item(),
        "positive_std": pos.std().item(),
        "negative_std": neg.std().item(),
        "gap": (pos.mean() - neg.mean()).item(),
        "local_loss": total_loc / total_samples,
        "ap_loss": total_ap / total_samples,
        "auc": auc,
        "top1" : moco_top1,
        "mrr": mrr_score,
        "map": map_score
    }


@torch.no_grad()
def validate_vit_moco_one_epoch(model, dataloader, device):
    model.eval()

    total_loss = 0.0
    total_samples = 0

    lambda_moco = 1.0
    lambda_local = 0.3
    lambda_ap = 1.0

    total_loc = 0.0
    total_ap = 0.0

    progress_bar = tqdm(dataloader, desc="Training", leave=False)

    for batch in progress_bar:
        x1_work_id = batch["src_work_id"].to(device)
        x2_work_id = batch["pos_work_id"].to(device)

        x1_perf_id = batch["src_label"].to(device)
        x2_perf_id = batch["pos_label"].to(device)

        x1 = batch["orig_inp"].to(device)
        x2 = batch["pos_inp"].to(device)
        neg_x2 = batch["neg_inp"].to(device)

        output = model(x1, x2)
        outputn = model(x1, neg_x2)

        # 1. flatten to have same pair on the batch
        query_pred = torch.cat([output["x1_query_pred"], output["x2_query_pred"]], dim=0) # [2B, D]
        key_feature = torch.cat([output["x1_key_feature"], output["x2_key_feature"]], dim=0) # [2B, D]

        work_ids = torch.cat([x1_work_id, x2_work_id], dim=0) # [2B]
        perf_ids = torch.cat([x1_perf_id, x2_perf_id], dim=0) # [2B]

        # 2. Identify if there are same work within the batch
        same_work = (work_ids[:, None] == work_ids[None, :]) # [2B, 2B]
        same_perf = (perf_ids[:, None] == perf_ids[None, :]) & same_work

        positive_mask = same_work & ~same_perf # positive: boolean id [2B]
        negative_mask = ~same_work # negative
        candidate_mask = ~same_perf # remove exactly same song

        query_pred = F.normalize(query_pred, dim=-1)
        key_feature = F.normalize(key_feature, dim=-1)

        sim_matrix = query_pred @ key_feature.T # [2B, 2B]

        pos = sim_matrix[positive_mask]
        neg = sim_matrix[negative_mask]

        # smooth AP loss
        embeddings = torch.cat([output["x1_glob_feature"], output["x2_glob_feature"]], dim=0) # [2B, D]
        work_ids = torch.cat([x1_work_id, x2_work_id], dim=0)

        ap_loss = smooth_ap_loss(embeddings, work_ids, temperature=0.01)

        # Top 1
        top1_matrix = sim_matrix.masked_fill(~candidate_mask, float("-inf"))

        pred = top1_matrix.argmax(dim=1) # [2B]
        row_idx = torch.arange(sim_matrix.size(0), device=sim_matrix.device)
        moco_top1 = positive_mask[row_idx, pred].float().mean()

        # loss = supervised_moco_loss(
        #     q=query_pred,
        #     k=key_feature,
        #     work_ids=work_ids,
        #     perf_ids=perf_ids,
        #     temperature=0.2,
        # )

        moco_loss = supervised_moco_loss(
            q=query_pred,
            k=key_feature,
            work_ids=work_ids,
            perf_ids=perf_ids,
            temperature=0.2,
        )

        pos_sim = output["pair_sim"]
        neg_sim = outputn["pair_sim"]
        neg_out = model.encoder_q.encode_one(neg_x2)
        # neg_local = neg_out["local_feature"]
        # _, _, neg_sim = model.encoder_q.pair_projection(
        #     output["x1_local_feature"],
        #     neg_local
        # )
        pos_local_score, _, _ = best_diagonal_window(pos_sim, window=40)
        neg_local_score, _, _ = best_diagonal_window(neg_sim, window=40)
        local_loss = F.relu(neg_local_score - pos_local_score + 0.2).mean()

        loss = lambda_moco * moco_loss + lambda_local * local_loss + lambda_ap * ap_loss

        batch_size = work_ids.size(0) # [2B]
        total_loss += loss.item() * batch_size
        total_samples += batch_size

        total_loc += local_loss.item() * batch_size # local loss
        total_ap += ap_loss.item() * batch_size # ap_loss

        # retrieval metrics
        retrieval_sim = sim_matrix.masked_fill(~candidate_mask, float("-inf"))

        # Calculate AUC
        valid_pair_mask = candidate_mask

        auc_scores = sim_matrix[valid_pair_mask]
        auc_labels = positive_mask[valid_pair_mask].long()

        mrr_score, map_score = get_moco_mrr(retrieval_sim, positive_mask)

        auc = roc_auc_score(auc_labels.detach().cpu().numpy(), auc_scores.detach().cpu().numpy(),)

    return {
        "loss": total_loss / total_samples,
        "positive_mean": pos.mean().item(),
        "negative_mean": neg.mean().item(),
        "positive_std": pos.std().item(),
        "negative_std": neg.std().item(),
        "gap": (pos.mean() - neg.mean()).item(),
        "local_loss": total_loc / total_samples,
        "ap_loss": total_ap / total_samples,
        "auc": auc,
        "top1" : moco_top1,
        "mrr": mrr_score,
        "map": map_score
    }


@torch.no_grad()
def evaluate_vit_moco_pn(model, dataloader, device):
    model.eval()

    total_loss = 0.0
    total_samples = 0

    lambda_moco = 1.0
    lambda_local = 0.3
    lambda_ap = 1.0

    total_loc = 0.0
    total_ap = 0.0

    progress_bar = tqdm(dataloader, desc="Training", leave=False)

    for batch in progress_bar:
        x1_work_id = batch["src_work_id"].to(device)
        x2_work_id = batch["pos_work_id"].to(device)

        x1_perf_id = batch["src_label"].to(device)
        x2_perf_id = batch["pos_label"].to(device)

        x1 = batch["orig_inp"].to(device)
        x2 = batch["pos_inp"].to(device)
        neg_x2 = batch["neg_inp"].to(device)

        output = model(x1, x2)
        outputn = model(x1, neg_x2)

        # 1. flatten to have same pair on the batch
        query_pred = torch.cat([output["x1_query_pred"], output["x2_query_pred"]], dim=0) # [2B, D]
        key_feature = torch.cat([output["x1_key_feature"], output["x2_key_feature"]], dim=0) # [2B, D]

        work_ids = torch.cat([x1_work_id, x2_work_id], dim=0) # [2B]
        perf_ids = torch.cat([x1_perf_id, x2_perf_id], dim=0) # [2B]

        # 2. Identify if there are same work within the batch
        same_work = (work_ids[:, None] == work_ids[None, :]) # [2B, 2B]
        same_perf = (perf_ids[:, None] == perf_ids[None, :]) & same_work

        positive_mask = same_work & ~same_perf # positive: boolean id [2B]
        negative_mask = ~same_work # negative
        candidate_mask = ~same_perf # remove exactly same song

        query_pred = F.normalize(query_pred, dim=-1)
        key_feature = F.normalize(key_feature, dim=-1)

        sim_matrix = query_pred @ key_feature.T # [2B, 2B]

        pos = sim_matrix[positive_mask]
        neg = sim_matrix[negative_mask]

        # smooth AP loss
        embeddings = torch.cat([output["x1_glob_feature"], output["x2_glob_feature"]], dim=0) # [2B, D]
        work_ids = torch.cat([x1_work_id, x2_work_id], dim=0)

        ap_loss = smooth_ap_loss(embeddings, work_ids, temperature=0.01)

        # Top 1
        top1_matrix = sim_matrix.masked_fill(~candidate_mask, float("-inf"))

        pred = top1_matrix.argmax(dim=1) # [2B]
        row_idx = torch.arange(sim_matrix.size(0), device=sim_matrix.device)
        moco_top1 = positive_mask[row_idx, pred].float().mean()

        # loss = supervised_moco_loss(
        #     q=query_pred,
        #     k=key_feature,
        #     work_ids=work_ids,
        #     perf_ids=perf_ids,
        #     temperature=0.2,
        # )

        moco_loss = supervised_moco_loss(
            q=query_pred,
            k=key_feature,
            work_ids=work_ids,
            perf_ids=perf_ids,
            temperature=0.2,
        )

        pos_sim = output["pair_sim"]
        neg_sim = outputn["pair_sim"]
        # neg_out = model.encoder_q.encode_one(neg_x2)
        # neg_local = neg_out["local_feature"]
        # _, _, neg_sim = model.encoder_q.pair_projection(
        #     output["x1_local_feature"],
        #     neg_local
        # )
        pos_local_score, _, _ = best_diagonal_window(pos_sim, window=40)
        neg_local_score, _, _ = best_diagonal_window(neg_sim, window=40)
        local_loss = F.relu(neg_local_score - pos_local_score + 0.2).mean()

        loss = lambda_moco * moco_loss + lambda_local * local_loss + lambda_ap * ap_loss

        batch_size = work_ids.size(0) # [2B]
        total_loss += loss.item() * batch_size
        total_samples += batch_size

        total_loc += local_loss.item() * batch_size # local loss
        total_ap += ap_loss.item() * batch_size # ap loss

        # retrieval metrics
        retrieval_sim = sim_matrix.masked_fill(~candidate_mask, float("-inf"))

        # Calculate AUC
        valid_pair_mask = candidate_mask

        auc_scores = sim_matrix[valid_pair_mask]
        auc_labels = positive_mask[valid_pair_mask].long()

        mrr_score, map_score = get_moco_mrr(retrieval_sim, positive_mask)

        auc = roc_auc_score(auc_labels.detach().cpu().numpy(), auc_scores.detach().cpu().numpy(),)

    return {
        "loss": total_loss / total_samples,
        "positive_mean": pos.mean().item(),
        "negative_mean": neg.mean().item(),
        "positive_std": pos.std().item(),
        "negative_std": neg.std().item(),
        "gap": (pos.mean() - neg.mean()).item(),
        "local_loss": total_loc / total_samples,
        "ap_loss": total_ap / total_samples,
        "auc": auc,
        "top1" : moco_top1,
        "mrr": mrr_score,
        "map": map_score
    }


@torch.no_grad()
def evaluate_vit_moco(model, dataloader, device):
    model.eval()

    total_loss = 0.0
    total_samples = 0

    progress_bar = tqdm(dataloader, desc="Training", leave=False)

    for batch in progress_bar:
        x1_work_id = batch["work_id"].to(device)
        x2_work_id = batch["work_id"].to(device)

        x1_perf_id = batch["src_perf_id"].to(device)
        x2_perf_id = batch["tgt_perf_id"].to(device)

        x1 = batch["orig_inp"].to(device)
        x2 = batch["cover_inp"].to(device)

        output = model(x1, x2)

        # 1. flatten to have same pair on the batch
        query_pred = torch.cat([output["x1_query_pred"], output["x2_query_pred"]], dim=0) # [2B, D]
        key_feature = torch.cat([output["x1_key_feature"], output["x2_key_feature"]], dim=0) # [2B, D]

        work_ids = torch.cat([x1_work_id, x2_work_id], dim=0) # [2B]
        perf_ids = torch.cat([x1_perf_id, x2_perf_id], dim=0) # [2B]

        # 2. Identify if there are same work within the batch
        same_work = (work_ids[:, None] == work_ids[None, :]) # [2B, 2B]
        same_perf = (perf_ids[:, None] == perf_ids[None, :]) & same_work

        positive_mask = same_work & ~same_perf # positive: boolean id [2B]
        negative_mask = ~same_work # negative
        candidate_mask = ~same_perf # remove exactly same song

        query_pred = F.normalize(query_pred, dim=-1)
        key_feature = F.normalize(key_feature, dim=-1)

        sim_matrix = query_pred @ key_feature.T # [2B, 2B]

        pos = sim_matrix[positive_mask]
        neg = sim_matrix[negative_mask]

        # Top 1
        top1_matrix = sim_matrix.masked_fill(~candidate_mask, float("-inf"))

        pred = top1_matrix.argmax(dim=1) # [2B]
        row_idx = torch.arange(sim_matrix.size(0), device=sim_matrix.device)
        moco_top1 = positive_mask[row_idx, pred].float().mean()

        loss = supervised_moco_loss(
            q=query_pred,
            k=key_feature,
            work_ids=work_ids,
            perf_ids=perf_ids,
            temperature=0.2,
        )

        batch_size = work_ids.size(0) # [2B]
        total_loss += loss.item() * batch_size
        total_samples += batch_size

        # retrieval metrics
        retrieval_sim = sim_matrix.masked_fill(~candidate_mask, float("-inf"))

        # Calculate AUC
        valid_pair_mask = candidate_mask

        auc_scores = sim_matrix[valid_pair_mask]
        auc_labels = positive_mask[valid_pair_mask].long()

        mrr_score, map_score = get_moco_mrr(retrieval_sim, positive_mask)

        auc = roc_auc_score(auc_labels.detach().cpu().numpy(), auc_scores.detach().cpu().numpy(),)

    return {
        "loss": total_loss / total_samples,
        "positive_mean": pos.mean().item(),
        "negative_mean": neg.mean().item(),
        "positive_std": pos.std().item(),
        "negative_std": neg.std().item(),
        "gap": (pos.mean() - neg.mean()).item(),
        "auc": auc,
        "top1" : moco_top1,
        "mrr": mrr_score,
        "map": map_score
    }


@torch.no_grad()
def evaluate_moco_datacos(model, dataloader, device, top_k=10):
    model.eval()

    total_loss = 0.0
    total_samples = 0

    all_query_pred = []
    all_key_feature = []
    all_work_ids = []
    all_perf_ids = []

    progress_bar = tqdm(dataloader, desc="Evaluation", leave=False)

    for batch in progress_bar:
        x1_work_id = batch["src_work_id"].to(device)
        x2_work_id = batch["pos_work_id"].to(device)

        x1_perf_id = batch["src_label"].to(device)
        x2_perf_id = batch["pos_label"].to(device)

        x1 = batch["orig_inp"].to(device)
        x2 = batch["pos_inp"].to(device)

        output = model(x1, x2)

        # [2B, D]
        query_pred = torch.cat(
            [output["x1_query_pred"], output["x2_query_pred"]],
            dim=0,
        )

        key_feature = torch.cat(
            [output["x1_key_feature"], output["x2_key_feature"]],
            dim=0,
        )

        # [2B]
        work_ids = torch.cat(
            [x1_work_id, x2_work_id],
            dim=0,
        )

        perf_ids = torch.cat(
            [x1_perf_id, x2_perf_id],
            dim=0,
        )

        # --------------------------------------------------
        # Loss is still calculated per batch
        # --------------------------------------------------
        query_pred_norm = F.normalize(query_pred, dim=-1)
        key_feature_norm = F.normalize(key_feature, dim=-1)

        loss = supervised_moco_loss(
            q=query_pred_norm,
            k=key_feature_norm,
            work_ids=work_ids,
            perf_ids=perf_ids,
            temperature=0.2,
        )

        batch_size = work_ids.size(0)

        total_loss += loss.item() * batch_size
        total_samples += batch_size

        # --------------------------------------------------
        # Only collect embeddings / IDs here
        # --------------------------------------------------
        all_query_pred.append(query_pred.detach().cpu())
        all_key_feature.append(key_feature.detach().cpu())

        all_work_ids.extend(work_ids.detach().cpu().tolist())
        all_perf_ids.extend(perf_ids.detach().cpu().tolist())

    # ======================================================
    # WHOLE Da-TACOS DATASET
    # ======================================================

    all_query_pred = torch.cat(all_query_pred, dim=0)
    all_key_feature = torch.cat(all_key_feature, dim=0)

    # ------------------------------------------------------
    # Remove duplicate performances caused by pair dataset
    # ------------------------------------------------------
    unique_query = []
    unique_key = []
    unique_work_ids = []
    unique_perf_ids = []

    seen_perf_ids = set()

    for i, perf_id in enumerate(all_perf_ids):
        if perf_id in seen_perf_ids:
            continue

        seen_perf_ids.add(perf_id)

        unique_query.append(all_query_pred[i])
        unique_key.append(all_key_feature[i])

        unique_work_ids.append(all_work_ids[i])
        unique_perf_ids.append(perf_id)

    query_pred = torch.stack(unique_query, dim=0).to(device)
    key_feature = torch.stack(unique_key, dim=0).to(device)

    work_ids = torch.tensor(
        unique_work_ids,
        device=device,
    )

    perf_ids = torch.tensor(
        unique_perf_ids,
        device=device,
    )

    print(
        "Da-TACOS retrieval:",
        len(all_perf_ids),
        "samples ->",
        len(unique_perf_ids),
        "unique performances",
    )

    # ======================================================
    # Normalize once
    # ======================================================

    query_pred = F.normalize(query_pred, dim=-1)
    key_feature = F.normalize(key_feature, dim=-1)

    # [N, N]
    sim_matrix = query_pred @ key_feature.T

    # ======================================================
    # Global masks
    # ======================================================

    same_work = (
        work_ids[:, None]
        ==
        work_ids[None, :]
    )

    same_perf = (
        perf_ids[:, None]
        ==
        perf_ids[None, :]
    ) & same_work

    # Same work, different performance
    positive_mask = same_work & ~same_perf

    # Different work
    negative_mask = ~same_work

    # Everything except exact same performance
    candidate_mask = ~same_perf

    # ======================================================
    # Positive / negative similarity statistics
    # ======================================================

    pos = sim_matrix[positive_mask]
    neg = sim_matrix[negative_mask]

    pos_mean = pos.mean().item()
    neg_mean = neg.mean().item()

    pos_std = (
        pos.std().item()
        if pos.numel() > 1
        else float("nan")
    )

    neg_std = (
        neg.std().item()
        if neg.numel() > 1
        else float("nan")
    )

    # ======================================================
    # Global retrieval matrix
    # ======================================================

    retrieval_sim = sim_matrix.masked_fill(
        ~candidate_mask,
        float("-inf"),
    )

    # Top-1 calculated once
    pred = retrieval_sim.argmax(dim=1)
    row_idx = torch.arange(retrieval_sim.size(0), device=device)
    moco_top1 = positive_mask[row_idx, pred].float().mean().item()

    # Top-k calculated once
    k = min(top_k, retrieval_sim.size(1))
    topk_idx = retrieval_sim.topk(k=k, dim=1).indices
    topk_positive = torch.gather(positive_mask, 1, topk_idx)
    moco_topk = topk_positive.any(dim=1).float().mean().item()

    # MRR / MAP calculated once
    mrr_score, map_score = get_moco_mrr(
        retrieval_sim,
        positive_mask,
    )

    if torch.is_tensor(mrr_score):
        mrr_score = mrr_score.item()

    if torch.is_tensor(map_score):
        map_score = map_score.item()

    # AUC calculated once
    auc_scores = sim_matrix[
        candidate_mask
    ]

    auc_labels = positive_mask[
        candidate_mask
    ].long()

    auc = roc_auc_score(
        auc_labels.cpu().numpy(),
        auc_scores.cpu().numpy(),
    )

    return {
        "loss": total_loss / total_samples,

        "positive_mean": pos_mean,
        "negative_mean": neg_mean,

        "gap": pos_mean - neg_mean,

        "positive_std": pos_std,
        "negative_std": neg_std,

        "auc": auc,

        "mrr": mrr_score,
        "map": map_score,
        "top1": moco_top1,
        "topk": moco_topk,
    }


def log_train_val(log_path, epoch, train_metrics, val_metrics):
    log_line = (
        "Epoch {:03d} | "
        "Train Loss: {:.4f} | Pos: {:.4f} | Neg: {:.4f} | Pos Std: {:.4f} | Neg Std: {:.4f} | Gap: {:.4f} | "
        "Local loss: {:.4f} | AP loss: {:.4f} | AUC: {:.4f} | Top1: {:.4f} | MRR: {:.4f} | MAP: {:.4f}\n"
        "Val Loss: {:.4f} | Pos: {:.4f} | Neg: {:.4f} | Pos Std: {:.4f} | Neg Std: {:.4f} | Gap: {:.4f} | "
        "Local loss: {:.4f} | AP loss: {:.4f} | AUC: {:.4f} | Top1: {:.4f} | MRR: {:.4f} | MAP: {:.4f}"
    ).format(
        epoch,
        train_metrics["loss"],
        train_metrics["positive_mean"],
        train_metrics["negative_mean"],
        train_metrics["positive_std"],
        train_metrics["negative_std"],
        train_metrics["gap"],
        train_metrics["local_loss"],
        train_metrics["ap_loss"],
        train_metrics["auc"],
        train_metrics["top1"],
        train_metrics["mrr"],
        train_metrics["map"],
        val_metrics["loss"],
        val_metrics["positive_mean"],
        val_metrics["negative_mean"],
        val_metrics["positive_std"],
        val_metrics["negative_std"],
        val_metrics["gap"],
        val_metrics["local_loss"],
        val_metrics["ap_loss"],
        val_metrics["auc"],
        val_metrics["top1"],
        val_metrics["mrr"],
        val_metrics["map"],
    )

    print(log_line)

    with open(log_path, "a") as f:
        f.write(log_line + "\n")


def log_eval_loc(log_path, dataset_name, metrics):
    log_line = (
        "{} | "
        "Loss: {:.4f} | Pos: {:.4f} | Neg: {:.4f} | Pos Std: {:.4f} | Neg Std: {:.4f} | Gap: {:.4f} | Local Loss: {:.4f} | AUC: {:.4f} | "
        "Top1: {:.4f} | MRR: {:.4f} | MAP: {:.4f}"
    ).format(
        dataset_name,
        metrics["loss"],
        metrics["positive_mean"],
        metrics["negative_mean"],
        metrics["positive_std"],
        metrics["negative_std"],
        metrics["gap"],
        metrics["local_loss"],
        metrics["auc"],
        metrics["top1"],
        metrics["mrr"],
        metrics["map"],
    )

    print(log_line)

    with open(log_path, "a") as f:
        f.write(log_line + "\n")


def log_eval(log_path, dataset_name, metrics):
    log_line = (
        "{} | "
        "Loss: {:.4f} | Pos: {:.4f} | Neg: {:.4f} | Pos Std: {:.4f} | Neg Std: {:.4f} | Gap: {:.4f} | AUC: {:.4f} | "
        "Top1: {:.4f} | MRR: {:.4f} | MAP: {:.4f}"
    ).format(
        dataset_name,
        metrics["loss"],
        metrics["positive_mean"],
        metrics["negative_mean"],
        metrics["positive_std"],
        metrics["negative_std"],
        metrics["gap"],
        metrics["auc"],
        metrics["top1"],
        metrics["mrr"],
        metrics["map"],
    )

    print(log_line)

    with open(log_path, "a") as f:
        f.write(log_line + "\n")
