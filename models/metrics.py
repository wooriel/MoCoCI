import numpy as np
import torch
import torch.nn.functional as F


# retrieval metric on train_vit
def retrieval_metrics(embeddings, work_ids, top_k=10):
    similarity = embeddings @ embeddings.T

    work_ids = np.asarray(work_ids)

    reciprocal_ranks = []
    average_precisions = []
    top1_hits = []
    topk_hits = []

    for i in range(len(embeddings)):
        scores = similarity[i].clone()

        scores[i] = -float("inf") # exclude query itself

        relevant = work_ids == work_ids[i]
        relevant[i] = False

        if relevant.sum() == 0: # no relevant cover -> skip as query (I would filter using decision)
            continue

        ranking = torch.argsort(scores, descending=True).cpu().numpy()
        ranked_relevant = relevant[ranking]
        relevant_ranks = np.where(ranked_relevant)[0] + 1

        reciprocal_ranks.append(1.0 / relevant_ranks[0]) # MRR

        precision_sum = 0.0 # AP
        num_found = 0

        for rank, is_relevant in enumerate(ranked_relevant, start=1):
            if is_relevant:
                num_found += 1
                precision_sum += num_found / rank

        ap = precision_sum / relevant.sum()
        average_precisions.append(ap)

        top1_hits.append(float(ranked_relevant[0])) # Top-1
        topk_hits.append(float(ranked_relevant[:top_k].any())) # Top-k

    return {
        "mrr": np.mean(reciprocal_ranks),
        "map": np.mean(average_precisions),
        "top1": np.mean(top1_hits),
        "top{}".format(top_k): np.mean(topk_hits),
        "num_queries": len(reciprocal_ranks),
    }


# train_align
def retrieval_metrics_val(embeddings, work_ids, performance_ids, top_k=10):
    if isinstance(embeddings, list):
        embeddings = torch.cat(embeddings, dim=0)

    work_ids = np.asarray(work_ids)
    performance_ids = np.asarray(performance_ids)

    assert len(embeddings) == len(work_ids) == len(performance_ids)

    # Keep each performance only once
    _, unique_indices = np.unique(performance_ids, return_index=True)
    unique_indices = np.sort(unique_indices)

    embeddings = embeddings[unique_indices]
    work_ids = work_ids[unique_indices]
    performance_ids = performance_ids[unique_indices]

    embeddings = F.normalize(embeddings, dim=-1)
    similarity = embeddings @ embeddings.T

    reciprocal_ranks = []
    average_precisions = []
    top1_hits = []
    topk_hits = []

    for i in range(len(embeddings)):
        scores = similarity[i].clone()
        scores[i] = -float("inf")

        relevant = (
            (work_ids == work_ids[i])
            & (performance_ids != performance_ids[i])
        )

        if relevant.sum() == 0:
            continue

        ranking = torch.argsort(scores, descending=True).cpu().numpy()

        ranked_relevant = relevant[ranking]
        relevant_ranks = np.where(ranked_relevant)[0] + 1

        reciprocal_ranks.append(1.0 / relevant_ranks[0])

        precision_sum = 0.0
        num_found = 0

        for rank, is_relevant in enumerate(ranked_relevant, start=1):
            if is_relevant:
                num_found += 1
                precision_sum += num_found / rank

        average_precisions.append(
            precision_sum / relevant.sum()
        )

        top1_hits.append(
            float(ranked_relevant[0])
        )

        topk_hits.append(
            float(ranked_relevant[:top_k].any())
        )

    return {
        "mrr": np.mean(reciprocal_ranks),
        "map": np.mean(average_precisions),
        "top1": np.mean(top1_hits),
        "topk": np.mean(topk_hits),
        "num_queries": len(reciprocal_ranks),
        "num_songs": len(embeddings),
    }


# train_moco
def get_moco_mrr(retrieval_sim, positive_mask):
    N = retrieval_sim.size(0)

    mrr_lst = []
    map_lst = []

    for i in range(N):
        scores = retrieval_sim[i] # [N]
        positives = positive_mask[i] # [N]
        num_pos = positives.sum()

        if positives.sum() == 0:
            continue

        ranking = torch.argsort(scores, descending=True)
        ranked_positive = positives[ranking] # select same coversong

        # MRR
        positive_ranks = torch.nonzero(ranked_positive, as_tuple=False).squeeze(1)
        first_rank = positive_ranks[0] + 1  # 1-based
        mrr_lst.append(1.0 / first_rank.float())

        # MAP
        cumulative_hits = torch.cumsum(ranked_positive.float(), dim=0)
        ranks = torch.arange(1, N + 1, device=scores.device, dtype=torch.float)
        precision_at_k = cumulative_hits / ranks
        ap = (precision_at_k * ranked_positive).sum() / num_pos.float()
        map_lst.append(ap)
        
    mrr_score = torch.stack(mrr_lst).mean()
    map_score = torch.stack(map_lst).mean()

    return mrr_score, map_score