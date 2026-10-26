import math
import torch
import torch.nn.functional as F


def gradient_cosine(grads_a, grads_b):
    flat_a = []
    flat_b = []

    for ga, gb in zip(grads_a, grads_b):
        # Only compare parameters that receive gradients
        # from BOTH losses
        if ga is not None and gb is not None:
            flat_a.append(ga.detach().reshape(-1))
            flat_b.append(gb.detach().reshape(-1))

    if len(flat_a) == 0:
        return float("nan")

    flat_a = torch.cat(flat_a)
    flat_b = torch.cat(flat_b)

    # Protect against zero gradient
    if flat_a.norm() == 0 or flat_b.norm() == 0:
        return float("nan")

    return F.cosine_similarity(
        flat_a.unsqueeze(0),
        flat_b.unsqueeze(0),
        dim=1
    ).item()


def get_loss_gradients(loss, module):
    params = [
        p for p in module.parameters()
        if p.requires_grad
    ]

    grads = torch.autograd.grad(
        loss,
        params,
        retain_graph=True,
        allow_unused=True
    )

    return grads


def gradient_norm(grads):
    valid = [
        g.detach().reshape(-1)
        for g in grads
        if g is not None
    ]

    if len(valid) == 0:
        return float("nan")

    flat = torch.cat(valid)
    return flat.norm().item()


def get_gradient_stats(module):
    grad_norms = []
    grad_means = []
    grad_maxes = []
    none_count = 0
    total_count = 0

    for name, param in module.named_parameters():
        if not param.requires_grad:
            continue

        total_count += 1

        if param.grad is None:
            none_count += 1
            continue

        grad = param.grad.detach()

        grad_norms.append(grad.norm().item())
        grad_means.append(grad.abs().mean().item())
        grad_maxes.append(grad.abs().max().item())

    if len(grad_norms) == 0:
        return {
            "grad_norm": 0.0,
            "grad_mean": 0.0,
            "grad_max": 0.0,
            "none_ratio": none_count / max(total_count, 1),
        }

    return {
        "grad_norm": sum(grad_norms) / len(grad_norms),
        "grad_mean": sum(grad_means) / len(grad_means),
        "grad_max": max(grad_maxes),
        "none_ratio": none_count / total_count,
    }


def safe_mean(values):
    values = [
        v for v in values
        if not math.isnan(v)
    ]

    if len(values) == 0:
        return float("nan")

    return sum(values) / len(values)