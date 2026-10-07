from tqdm import tqdm
from models.f_losses import cosine_pair_loss, contrastive_loss, triplet_loss, retrieval_metrics
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from graph import glob_pair_heatmap


def train_vit_one_epoch(model, dataloader, optimizer, device, margin=0.3):
    model.train()

    total_loss = 0.0
    total_samples = 0

    all_pos_sim = []
    all_neg_sim = []

    progress_bar = tqdm(dataloader, desc="Training", leave=False)

    for batch in progress_bar:
        src = batch["orig_inp"].to(device)
        tgt = batch["cover_inp"].to(device)

        pair_label = batch["decision"].to(device=device, dtype=torch.float32,)

        optimizer.zero_grad()
        outputs = model(src, tgt)

        cosine_sim = outputs["cosine_similarity"]

        losses = cosine_pair_loss(
            cosine_similarity=cosine_sim,
            pair_label=pair_label,
            margin=margin,
        )

        cont_loss = contrastive_loss(
            src=outputs["src_feature"],
            tgt=outputs["tgt_feature"],
            label=pair_label,
            margin=1.0
        )

        loss = cont_loss
        loss.backward()
        optimizer.step()

        batch_size = (pair_label.size(0))

        total_loss += (loss.item() * batch_size)
        total_samples += batch_size

        pos_mask = pair_label == 1
        neg_mask = pair_label == 0

        if pos_mask.any():
            all_pos_sim.append(cosine_sim[pos_mask].detach().cpu())

        if neg_mask.any():
            all_neg_sim.append(cosine_sim[neg_mask].detach().cpu())

    if len(all_pos_sim) > 0:
        pos_mean = torch.cat(all_pos_sim).mean().item()
    else:
        pos_mean = float("nan")

    if len(all_neg_sim) > 0:
        neg_mean = torch.cat(all_neg_sim).mean().item()
    else:
        neg_mean = float("nan")

    return {
        "loss":total_loss / total_samples,
        "positive_cosine_mean": pos_mean,
        "negative_cosine_mean": neg_mean,
        "cosine_gap": pos_mean - neg_mean,
    }


@torch.no_grad()
def validate_vit_one_epoch(model, dataloader, device, margin=0.3, top_k=10):

    model.eval()

    total_loss = 0.0
    total_samples = 0

    all_cosine_sim = []
    all_labels = []
    all_embeddings = [] # for MRR
    all_work_ids = []

    progress_bar = tqdm(dataloader, desc="Validation", leave=False)

    for batch in progress_bar:

        src = batch["orig_inp"].to(device)
        tgt = batch["cover_inp"].to(device)

        pair_label = batch["decision"].to(device=device, dtype=torch.float32)
        pos_mask = pair_label == 1

        outputs = model(src, tgt)
        if pos_mask.any():
            work_id = batch["work_id"][pos_mask.cpu().tolist()]
            all_embeddings.append(outputs["src_feature"][pos_mask])
            all_embeddings.append(outputs["tgt_feature"][pos_mask])
            all_work_ids.extend(work_id.cpu().tolist())
            all_work_ids.extend(work_id.cpu().tolist())

        cosine_sim = outputs["cosine_similarity"]

        losses = cosine_pair_loss(
            cosine_similarity=cosine_sim,
            pair_label=pair_label,
            margin=margin,
        )

        cont_loss = contrastive_loss(
            src=outputs["src_feature"],
            tgt=outputs["tgt_feature"],
            label=pair_label,
            margin=1.0
        )

        loss = cont_loss # losses["loss"]
        # pos_loss = losses["positive_loss"]
        # neg_loss = losses["negative_loss"]

        batch_size = (pair_label.size(0))

        total_loss += (loss.item() * batch_size)
        total_samples += batch_size

        all_cosine_sim.append(cosine_sim.cpu())
        all_labels.append(pair_label.long().cpu())

    all_cosine_sim = torch.cat(all_cosine_sim, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    pos_mask = all_labels == 1
    neg_mask = all_labels == 0

    pos_sim = all_cosine_sim[pos_mask]
    neg_sim = all_cosine_sim[neg_mask]

    positive_mean = (pos_sim.mean().item() if pos_mask.any() else float("nan"))
    negative_mean = (neg_sim.mean().item() if neg_mask.any() else float("nan"))

    if (pos_mask.any() and neg_mask.any()):
        roc_auc = roc_auc_score(
            all_labels.numpy(),
            all_cosine_sim.numpy(),
        )
    else:
        roc_auc = float("nan")

    all_embeddings = torch.cat(all_embeddings, dim=0)
    ret_metrics = retrieval_metrics(all_embeddings, all_work_ids, top_k=top_k) # calculate MRR | MAP
    
    return {
        "loss": total_loss / total_samples,
        "positive_cosine_mean": positive_mean,
        "negative_cosine_mean": negative_mean,
        "cosine_gap": positive_mean - negative_mean,
        "roc_auc": roc_auc,
        "mrr": ret_metrics["mrr"],
        "map": ret_metrics["map"],
        "top1" : ret_metrics["top1"],
        "topk": ret_metrics["top{}".format(top_k)]
    }


@torch.no_grad()
def evaluate_vit(model, dataloader, device, save_path, epoch_name, margin=0.3, top_k=10):
    model.eval()
    total_loss = 0.0
    total_samples = 0

    all_cosine_sim = []
    all_labels = []
    all_embeddings = [] # for MRR
    all_work_ids = []

    all_src_emb = []
    all_tgt_emb = []

    for batch in tqdm(dataloader, desc="Evaluation", leave=False):
        src = batch["orig_inp"].to(device)
        tgt = batch["cover_inp"].to(device)
        work_id = batch["work_id"]

        pair_label = batch["decision"].to(device=device, dtype=torch.float32)
        pos_mask = pair_label == 1

        outputs = model(src, tgt)
        if pos_mask.any():
            work_id = batch["work_id"][pos_mask.cpu().tolist()]
            all_embeddings.append(outputs["src_feature"][pos_mask])
            all_embeddings.append(outputs["tgt_feature"][pos_mask])
            src_emb = outputs["src_feature"][pos_mask]
            tgt_emb = outputs["tgt_feature"][pos_mask]
            all_src_emb.append(src_emb)
            all_tgt_emb.append(tgt_emb)
            all_work_ids.extend(work_id.cpu().tolist())
            all_work_ids.extend(work_id.cpu().tolist())

            # local_src = outputs["src_local"][pos_mask][0]
            # local_tgt = outputs["tgt_local"][pos_mask][0]
            # print(local_src.shape)
            # print(local_tgt.shape)
            # pair_sim = outputs["pair_similarity"][pos_mask][0]
            # print(pair_sim.shape)
            # pair_heatmap(save_path, work_id[0], epoch_name, pair_sim)

        cosine_sim = outputs["cosine_similarity"]
        losses = cosine_pair_loss(cosine_similarity=cosine_sim, pair_label=pair_label, margin=margin)
        cont_loss = contrastive_loss(
            src=outputs["src_feature"],
            tgt=outputs["tgt_feature"],
            label=pair_label,
            margin=1.0
        )
        loss = cont_loss
        # loss = losses["loss"]
        pos_loss = losses["positive_loss"]
        neg_loss = losses["negative_loss"]
        batch_size = pair_label.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size
        all_cosine_sim.append(cosine_sim.cpu())
        all_labels.append(pair_label.long().cpu())

    all_cosine_sim = torch.cat(all_cosine_sim)
    all_labels = torch.cat(all_labels)
    pos_mask = all_labels == 1
    neg_mask = all_labels == 0
    pos_sim = all_cosine_sim[pos_mask]
    neg_sim = all_cosine_sim[neg_mask]
    pos_mean = pos_sim.mean().item() if pos_mask.any() else float("nan")
    neg_mean = neg_sim.mean().item() if neg_mask.any() else float("nan")
    pos_std = pos_sim.std().item() if len(pos_sim) > 1 else float("nan")
    neg_std = neg_sim.std().item() if len(neg_sim) > 1 else float("nan")
    if pos_mask.any() and neg_mask.any():
        auc = roc_auc_score(all_labels.numpy(), all_cosine_sim.numpy())
    else:
        auc = float("nan")
    print("positive cosine:", pos_mean, "+/-", pos_std)
    print("negative cosine:", neg_mean, "+/-", neg_std)
    print("cosine gap:", pos_mean - neg_mean)
    print("cosine min/max:", all_cosine_sim.min().item(), all_cosine_sim.max().item())
    print("ROC-AUC:", auc)
    
    all_embeddings = torch.cat(all_embeddings, dim=0)
    all_src_emb = torch.cat(all_src_emb, dim=0)
    all_tgt_emb = torch.cat(all_tgt_emb, dim=0)
    ret_metrics = retrieval_metrics(all_embeddings, all_work_ids, top_k=top_k) # calculate MRR | MAP

    # glob_pair_heatmap(save_path, epoch_name, all_src_emb, all_tgt_emb)

    return {
        "loss": total_loss / total_samples,
        "positive_cosine_mean": pos_mean,
        "negative_cosine_mean": neg_mean,
        "positive_cosine_std": pos_std,
        "negative_cosine_std": neg_std,
        "cosine_gap": pos_mean - neg_mean,
        "roc_auc": auc,
        "mrr": ret_metrics["mrr"],
        "map": ret_metrics["map"],
        "top1" : ret_metrics["top1"],
        "topk": ret_metrics["top{}".format(top_k)]
    }