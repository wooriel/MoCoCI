import torch
import torch.nn.functional as F


# train_vit
def cosine_pair_loss(cosine_similarity, pair_label, margin=0.3):
    pair_label = pair_label.long()

    pos_mask = pair_label == 1
    neg_mask = pair_label == 0

    if pos_mask.any():
        pos_loss = (
            1.0 - cosine_similarity[pos_mask]
        ).mean()
    else:
        pos_loss = cosine_similarity.new_tensor(0.0)

    if neg_mask.any():
        neg_loss = F.relu(
            cosine_similarity[neg_mask] - margin
        ).mean()
    else:
        neg_loss = cosine_similarity.new_tensor(0.0)

    loss = pos_loss + neg_loss

    return {
        "loss": loss,
        "positive_loss": pos_loss,
        "negative_loss": neg_loss,
    }


def contrastive_loss(src, tgt, label, margin=1.0):
    dist = F.pairwise_distance(src, tgt)
    pos_loss = label * dist.pow(2)
    neg_loss = (1 - label) * F.relu(margin - dist).pow(2)
    return (pos_loss + neg_loss).mean()


# train_align (hpcp)
def triplet_loss(cos_pos, cos_neg, margin=0.1):
    return F.relu(cos_neg - cos_pos + margin).mean()


# train_align, train_moco: local score
def best_diagonal_window(sim, window=8, return_indices=False):
    B, N, M= sim.shape
    if window > N or window > M:
        raise ValueError(f"window={window} > N={N} or M={M}")
    
    sb, sn, sm = sim.stride()
    
    windows = sim.as_strided(
        size = (B,N-window+1,M-window+1,window),
        stride = (sb,sn,sm,sn+sm)
    )
    score_map = windows.mean(dim=-1)
    best_score, best_idx = score_map.flatten(1).max(dim=1)

    if not return_indices:
        return best_score, None, None
    
    n_tgt = M - window+1
    src_start = best_idx // n_tgt
    tgt_start = best_idx % n_tgt

    return best_score, src_start, tgt_start


# train_moco: moco_loss
def supervised_moco_loss(q, k, work_ids, perf_ids, temperature=0.2):
    q = F.normalize(q, dim=-1)
    k = F.normalize(k, dim=-1)

    logits = q @ k.T / temperature            # [2B, 2B]

    same_work = (work_ids[:, None] == work_ids[None, :])
    same_perf = ((perf_ids[:, None] == perf_ids[None, :]) & same_work)

    positive_mask = same_work & ~same_perf
    candidate_mask = ~same_perf

    # completely exclude exact same performance
    logits = logits.masked_fill(~candidate_mask, float("-inf"))

    log_prob = F.log_softmax(logits, dim=1)

    num_pos = positive_mask.sum(dim=1)
    valid = num_pos > 0

    # prevent 0 * -inf -> nan
    log_prob = torch.where(
        positive_mask,
        log_prob,
        torch.zeros_like(log_prob),
    )

    mean_log_prob_pos = (log_prob.sum(dim=1) / num_pos.clamp_min(1))

    return -mean_log_prob_pos[valid].mean()


# train_align, train_moco: ap_loss
def smooth_ap_loss(embeddings, work_ids, temperature=0.01):
    embeddings = F.normalize(embeddings, dim=-1)

    sim = embeddings @ embeddings.T
    B = sim.size(0)

    eye = torch.eye(B, dtype=torch.bool, device=sim.device)
    same_work = work_ids[:, None] == work_ids[None, :]

    positive_mask = same_work & ~eye
    candidate_mask = ~eye

    ap_list = []

    for i in range(B):
        scores = sim[i]
        positives = positive_mask[i]

        if positives.sum() == 0:
            continue

        valid_scores = scores[candidate_mask[i]]
        valid_pos = positives[candidate_mask[i]]

        pos_scores = valid_scores[valid_pos]
        precision_list = []

        for p_idx, s_pos in enumerate(pos_scores):
            score_diff = (valid_scores - s_pos) / temperature

            soft_rank = 1.0 + torch.sigmoid(score_diff).sum()

            pos_diff = (pos_scores - s_pos) / temperature

            pos_self_mask = torch.ones_like(pos_scores, dtype=torch.bool)
            pos_self_mask[p_idx] = False

            soft_pos_rank = 1.0 + torch.sigmoid(
                pos_diff[pos_self_mask]
            ).sum()

            precision = soft_pos_rank / soft_rank
            precision_list.append(precision)

        ap_i = torch.stack(precision_list).mean()
        ap_list.append(ap_i)

    return 1.0 - torch.stack(ap_list).mean()