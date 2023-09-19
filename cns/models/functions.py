import torch
import torch.nn.functional as F
from ..midend.graph_gen import GraphData


def scaled_cosine_similarity_loss(pred, target):
    similarity = F.cosine_similarity(pred, target, dim=-1)
    loss = 1 - similarity
    return loss


def alpha_p_elu(x, alpha=1.0):
    return alpha + F.elu(x, alpha=alpha)


def reverse_alpha_p_elu(y: torch.Tensor, alpha=1.0, eps=1e-7):
    mask = y < alpha
    if mask.ndim == 0:  # is scalar
        x = torch.log(y/alpha + eps) if mask else (y - alpha)
    else:
        x = y - alpha
        x[mask] = torch.log(y[mask] / alpha + eps)
    return x


def postprocess(raw_pred, data: GraphData, post_scale: bool) -> torch.Tensor:
    vel_vec, vel_norm = raw_pred[0], raw_pred[1]
    if vel_norm is None:
        vel = vel_vec
    else:
        vel = vel_vec / (torch.norm(
            vel_vec, dim=-1, keepdim=True) + 1e-7) * \
            alpha_p_elu(vel_norm, alpha=1)
    
    if post_scale:
        vel = vel.clone()
        vel[:, :3] *= getattr(data, "tPo_norm").view(-1, 1)
    return vel


def objectives(raw_pred, data: GraphData, gt_si: bool, weight=1):
    vel_vec, vel_norm = raw_pred[0], raw_pred[1]
    gt = getattr(data, "vel_si") if gt_si else getattr(data, "vel")
    dir_loss = scaled_cosine_similarity_loss(vel_vec * weight, gt * weight).mean()

    if vel_norm is None:
        # we need to constraint the norm of vel_vec
        norm_loss = F.mse_loss(vel_vec * weight, gt * weight, reduction="mean")
    else:
        # norm is regressed separately
        gt_norm = torch.norm(gt, dim=-1, keepdim=True)
        norm_loss = F.mse_loss(
            vel_norm, 
            reverse_alpha_p_elu(gt_norm, eps=1e-5), 
            reduction="mean"
        # ) * 0.5
        )
    
    loss = dir_loss + norm_loss
    result = {
        "dir_loss": dir_loss.item(),
        "norm_loss": norm_loss.item(),
        "total_loss": loss.item()
    }
    return result, loss
