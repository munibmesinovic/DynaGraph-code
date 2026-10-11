import torch
import torch.nn as nn
import torch.nn.functional as F


def augment_time_shuffle(x):
    B, C, Feat, T = x.shape
    idx = torch.stack([torch.randperm(T, device=x.device) for _ in range(B)])
    return torch.gather(x, 3, idx.view(B, 1, 1, T).expand(-1, C, Feat, -1))


def node_keep_mask(batch_size, num_nodes, drop_rate, like):
    return torch.bernoulli(torch.full((batch_size, num_nodes), 1.0 - drop_rate, dtype=like.dtype, device=like.device))


def edge_keep_mask(num_windows, num_nodes, drop_rate, like):
    keep = torch.bernoulli(torch.full((num_windows, num_nodes, num_nodes), 1.0 - drop_rate,
                                      dtype=like.dtype, device=like.device))
    upper = torch.triu(keep, diagonal=1)
    return upper + upper.transpose(1, 2) + torch.eye(num_nodes, dtype=like.dtype, device=like.device)


def abs_correlation(x, eps=1e-6):
    xc = x - x.mean(dim=-1, keepdim=True)
    sd = xc.pow(2).mean(dim=-1).sqrt()
    moving = sd > eps
    z = xc / sd.clamp(min=eps).unsqueeze(-1)
    score = (z @ z.transpose(1, 2)).abs() / x.shape[-1]
    return score * (moving.unsqueeze(-1) & moving.unsqueeze(-2))


def comovement_graph(x, k, eps=1e-6):
    x = x.squeeze(1) if x.dim() == 4 else x
    score = abs_correlation(x, eps)
    eye = torch.eye(x.shape[1], dtype=torch.bool, device=x.device)
    score = score.masked_fill(eye, 0.0)
    pick = torch.zeros_like(score).scatter_(-1, score.topk(min(k, x.shape[1] - 1), dim=-1).indices, 1.0)
    pick = pick * (score > 0)
    return ((pick + pick.transpose(1, 2)) > 0).to(x.dtype)


def context_hours(num_hours, num_windows, context):
    w = num_hours // num_windows
    spans = []
    for t in range(num_windows):
        a = min(max(t * w + w // 2 - context // 2, 0), num_hours - context)
        spans.append((a, a + context))
    return spans


def comovement_window_graphs(x, k, num_windows, context):
    if context >= x.shape[-1]:
        return comovement_graph(x, k).unsqueeze(1).expand(-1, num_windows, -1, -1)
    spans = context_hours(x.shape[-1], num_windows, context)
    return torch.stack([comovement_graph(x[..., a:b], k) for a, b in spans], dim=1)


def nt_xent(anchor, positive, negative, tau):
    a = F.normalize(anchor, dim=-1)
    p = F.normalize(positive, dim=-1)
    n = F.normalize(negative, dim=-1)
    sim_pos = a @ p.t() / tau
    sim_neg = (a * n).sum(dim=-1, keepdim=True) / tau
    logits = torch.cat([sim_pos, sim_neg], dim=1)
    return (torch.logsumexp(logits, dim=1) - sim_pos.diagonal()).mean()


class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg, gamma_pos, clip, pos_weight, eps=1e-8):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps
        self.register_buffer("pos_weight", pos_weight)

    def forward(self, logits, targets):
        targets = targets.to(logits.dtype)
        probs = torch.sigmoid(logits)
        p_m = (probs - self.clip).clamp(min=0)
        pos_loss = targets * torch.pow(1 - probs, self.gamma_pos) * F.logsigmoid(logits)
        pos_loss = pos_loss * self.pos_weight.to(logits.device, logits.dtype).unsqueeze(0)
        neg_loss = (1 - targets) * torch.pow(p_m, self.gamma_neg) * torch.log((1 - p_m).clamp(min=self.eps))
        return (-(pos_loss + neg_loss)).mean()


class DynaGraphLoss(nn.Module):
    def __init__(self, cfg, pos_weight):
        super().__init__()
        lc = cfg["loss"]
        self.asl = AsymmetricLoss(lc["gamma_neg"], lc["gamma_pos"], lc["asl_clip"], pos_weight)
        self.alpha_contrast = lc["alpha_contrast"]
        self.lambda_reg = lc["lambda_reg"]
        self.mu_struct = lc["mu_struct"]
        self.vgae_weight = lc["vgae_weight"]
        self.beta_vgae = lc["beta_vgae"]
        self.struct_decay = lc["struct_decay"]
        self.struct_decay_start = lc["struct_decay_start"]
        self.kl_anneal_epochs = lc["kl_anneal_epochs"]
        self.vgae_context = lc["vgae_context"]
        self.tau = lc["tau"]
        self.node_drop = lc["node_drop"]
        self.edge_drop = lc["edge_drop"]

    def contrastive_loss(self, model, x_input, anchor):
        node_keep = node_keep_mask(x_input.shape[0], model.num_features, self.node_drop, x_input)
        edge_keep = edge_keep_mask(model.num_groups, model.num_features, self.edge_drop, x_input)
        positive = model(x_input, node_keep=node_keep, edge_keep=edge_keep)["z_contrast"]
        negative = model(augment_time_shuffle(x_input))["z_contrast"]
        return nt_xent(anchor, positive, negative, self.tau)

    def regularization_loss(self, A):
        return (A ** 2).mean()

    def structural_loss(self, A, z):
        dist_sq = ((z.unsqueeze(-2) - z.unsqueeze(-3)) ** 2).sum(dim=-1)
        return (A * dist_sq).sum() / A.sum().clamp(min=1e-8)

    def vgae_loss(self, A_recon, target_graph, mu, logvar, current_epoch, eps=1e-8):
        n = target_graph.shape[-1]
        off = ~torch.eye(n, dtype=torch.bool, device=target_graph.device)
        recon, kl = [], []
        for t in range(len(mu)):
            target = (target_graph[:, t] > 0).to(A_recon[t].dtype).detach()
            n_pos = target[..., off].sum()
            w = (off.sum() * (target.numel() // (n * n)) - n_pos) / n_pos.clamp(min=1.0)
            bce = -(w * target * torch.log(A_recon[t].clamp(min=eps))
                    + (1 - target) * torch.log((1 - A_recon[t]).clamp(min=eps)))
            recon.append(bce[:, off].mean())
            kl.append(-0.5 * torch.mean(1.0 + logvar[t] - mu[t].pow(2) - logvar[t].exp()))
        recon, kl = torch.stack(recon).mean(), torch.stack(kl).mean()
        beta_eff = self.beta_vgae * min(1.0, current_epoch / max(self.kl_anneal_epochs, 1))
        return recon + beta_eff * kl

    def forward(self, model_output, targets, model, x_input, current_epoch):
        l_focal = self.asl(model_output["logits"], targets)
        l_contrast = self.contrastive_loss(model, x_input, model_output["z_contrast"])
        A = model_output["A"]
        l_reg = self.regularization_loss(A)
        l_struct = self.structural_loss(A[:, -1], model_output["z"][-1])
        target_graph = comovement_window_graphs(x_input, model.graph_constructor.k, model.num_groups, self.vgae_context)
        l_vgae = self.vgae_loss(model_output["A_recon"], target_graph, model_output["mu"],
                                model_output["logvar"], current_epoch)

        mu_eff = self.mu_struct
        if current_epoch > self.struct_decay_start:
            mu_eff *= self.struct_decay ** (current_epoch - self.struct_decay_start)

        total = (
            l_focal
            + self.alpha_contrast * l_contrast
            + self.lambda_reg * l_reg
            + mu_eff * l_struct
            + self.vgae_weight * l_vgae
        )
        terms = {
            "focal": l_focal.item(),
            "contrastive": l_contrast.item(),
            "regularization": l_reg.item(),
            "structural": l_struct.item(),
            "mu_struct_eff": mu_eff,
            "vgae_total": l_vgae.item(),
            "total": total.item(),
        }
        return total, terms
