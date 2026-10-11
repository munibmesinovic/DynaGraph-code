import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter, init


class DynamicGraphConstructor(nn.Module):
    def __init__(self, num_nodes, k, rank, num_windows, embed_dim):
        super().__init__()
        self.num_nodes = num_nodes
        self.k = min(k, num_nodes - 1)
        self.num_windows = num_windows
        self.theta = Parameter(torch.empty(num_nodes, rank))
        self.psi = Parameter(torch.empty(num_nodes, rank))
        init.xavier_uniform_(self.theta)
        init.xavier_uniform_(self.psi)
        self.U = Parameter(torch.zeros(num_windows, num_nodes, rank))
        self.V = Parameter(torch.zeros(num_windows, num_nodes, rank))
        self.P = nn.Linear(embed_dim, rank, bias=False)
        self.Q = nn.Linear(embed_dim, rank, bias=False)
        init.zeros_(self.P.weight)
        init.zeros_(self.Q.weight)

    def probabilities(self, h):
        theta = self.theta + self.U + self.P(h).transpose(1, 2)
        psi = self.psi + self.V + self.Q(h).transpose(1, 2)
        return torch.sigmoid(theta @ psi.transpose(-1, -2))

    def forward(self, h):
        P = self.probabilities(h)
        eye = torch.eye(self.num_nodes, dtype=torch.bool, device=P.device)
        P = P.masked_fill(eye, 0.0)
        mask = torch.zeros_like(P).scatter_(-1, P.topk(self.k, dim=-1).indices, 1.0)
        A = P * mask
        return 0.5 * (A + A.transpose(-1, -2))


class InterpretabilityMatrix(nn.Module):
    def __init__(self, num_nodes):
        super().__init__()
        self.E = Parameter(torch.empty(num_nodes, num_nodes))
        init.uniform_(self.E, 0.0, 1.0)
        self.register_buffer("grad_accum", torch.zeros(num_nodes, num_nodes))

    def update_from_gradients(self, momentum=0.9):
        if self.E.grad is not None:
            self.grad_accum = momentum * self.grad_accum + (1.0 - momentum) * self.E.grad.abs()

    def forward(self):
        normed = self.grad_accum / (self.grad_accum.max() + 1e-8)
        return torch.sigmoid(self.E) * (1.0 + normed)


def fuse_graph(A, E):
    I = torch.eye(A.shape[-1], dtype=A.dtype, device=A.device)
    M = (A + E) * (I + A)
    deg_inv_sqrt = M.sum(dim=-1).clamp(min=1e-8).pow(-0.5)
    return deg_inv_sqrt.unsqueeze(-1) * M * deg_inv_sqrt.unsqueeze(-2)


class TemporalEncoder(nn.Module):
    def __init__(self, hidden_dim, num_layers, dropout, num_windows):
        super().__init__()
        self.num_windows = num_windows
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name or "weight_hh" in name:
                init.orthogonal_(param)
            elif "bias" in name:
                init.zeros_(param)

    def forward(self, x):
        B, N, T = x.shape
        S = self.num_windows
        L = T // S
        out, _ = self.lstm(x.reshape(B * N * S, L, 1))
        return out.reshape(B, N, S, L, -1).mean(dim=3)


class GINLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.eps = Parameter(torch.zeros(1))
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x, A):
        return self.mlp((1.0 + self.eps) * x + torch.matmul(A, x))


class GINEncoder(nn.Module):
    def __init__(self, input_dim, dims):
        super().__init__()
        layers = []
        prev = input_dim
        for d in dims:
            layers.append(GINLayer(prev, d))
            prev = d
        self.layers = nn.ModuleList(layers)

    def forward(self, x, A):
        for layer in self.layers:
            x = layer(x, A)
        return x


class GraphPooling(nn.Module):
    def __init__(self, in_dim, num_clusters):
        super().__init__()
        self.assign_net = nn.Sequential(
            nn.Linear(in_dim, max(in_dim // 4, 1)),
            nn.ReLU(),
            nn.Linear(max(in_dim // 4, 1), num_clusters),
        )
        self.attn = nn.Linear(in_dim, 1)

    def forward(self, X, G):
        M = F.softmax(self.assign_net(X), dim=-1)
        X_pool = M.transpose(1, 2) @ X
        G_pool = M.transpose(1, 2) @ G @ M
        X_pool = X_pool + (G_pool / G_pool.sum(-1, keepdim=True).clamp(min=1e-8)) @ X_pool
        attn_w = F.softmax(self.attn(X_pool), dim=1)
        return (attn_w * X_pool).sum(dim=1)


class MLPClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.net(x)


class DynaGraph(nn.Module):
    def __init__(
        self,
        num_features,
        num_classes,
        num_groups,
        lstm_hidden,
        lstm_layers,
        gin_dims,
        latent_dim,
        k_neighbors,
        graph_rank,
        dropout,
        pool_clusters,
        node_readout_dim,
        alpha_init,
    ):
        super().__init__()
        self.num_features = num_features
        self.num_groups = num_groups
        self.latent_dim = latent_dim

        embed_dim = 2 * lstm_hidden
        self.temporal_encoder = TemporalEncoder(lstm_hidden, lstm_layers, dropout, num_groups)
        self.node_embedding = Parameter(0.02 * torch.randn(num_features, embed_dim))
        self.current_state_weight = Parameter(0.02 * torch.randn(num_features, embed_dim))
        self.current_state_bias = Parameter(torch.zeros(num_features, embed_dim))

        self.graph_constructor = DynamicGraphConstructor(num_features, k_neighbors, graph_rank, num_groups, embed_dim)
        self.interp_matrix = InterpretabilityMatrix(num_features)

        self.gin_encoder = GINEncoder(embed_dim, gin_dims)
        self.mu_head = GINLayer(gin_dims[-1], latent_dim)
        self.logvar_head = GINLayer(gin_dims[-1], latent_dim)
        self.lstm_residual_proj = nn.Linear(embed_dim, latent_dim)

        alpha_logit = math.log(max(alpha_init, 1e-6) / max(1e-6, 1.0 - alpha_init))
        self.prop_alpha_logit = Parameter(torch.tensor(alpha_logit))
        self.prop_proj = nn.Linear(gin_dims[-1], embed_dim)

        total_feat_dim = 2 * latent_dim * num_groups
        self.graph_pool = GraphPooling(total_feat_dim, pool_clusters)
        self.contrast_proj = nn.Sequential(nn.Linear(total_feat_dim, 64), nn.ReLU(), nn.Linear(64, 32))
        self.node_pool = nn.Sequential(nn.Linear(total_feat_dim, 128), nn.ReLU(), nn.Dropout(dropout))
        self.node_readout = nn.Linear(latent_dim * num_groups, node_readout_dim)
        self.classifier = MLPClassifier(128 + num_features * node_readout_dim, num_classes, dropout)

    def prop_alpha(self):
        return torch.sigmoid(self.prop_alpha_logit)

    def embed_nodes(self, x):
        return self.temporal_encoder(x) + self.node_embedding[None, :, None, :]

    def encode_windows(self, h, G, x_last):
        alpha = self.prop_alpha()
        zs, mus, logvars, readout = [], [], [], []
        prev_z = None
        for t in range(self.num_groups):
            h_t = h[:, :, t]
            if prev_z is not None:
                h_t = h_t + alpha * self.prop_proj(prev_z)
            if t == self.num_groups - 1:
                h_t = h_t + x_last.unsqueeze(-1) * self.current_state_weight + self.current_state_bias
            G_t = G[:, t]
            z_t = self.gin_encoder(h_t, G_t)
            mu_t = self.mu_head(z_t, G_t)
            logvar_t = self.logvar_head(z_t, G_t)
            prev_z = z_t
            zs.append(z_t)
            mus.append(mu_t)
            logvars.append(logvar_t)
            readout.append(torch.cat([mu_t, self.lstm_residual_proj(h_t)], dim=-1))
        return zs, mus, logvars, torch.cat(readout, dim=-1)

    def forward(self, x, node_keep=None, edge_keep=None):
        x = x.squeeze(1)
        if node_keep is not None:
            x = x * node_keep.unsqueeze(-1)

        h = self.embed_nodes(x)
        A = self.graph_constructor(h)
        if edge_keep is not None:
            A = A * edge_keep
        G = fuse_graph(A, self.interp_matrix())

        zs, mus, logvars, readout = self.encode_windows(h, G, x[:, :, -1])
        A_recon = []
        for mu_t, logvar_t in zip(mus, logvars):
            Z_t = mu_t
            if self.training:
                Z_t = mu_t + torch.randn_like(mu_t) * torch.exp(0.5 * logvar_t.clamp(-10.0, 10.0))
            A_recon.append(torch.sigmoid(Z_t @ Z_t.transpose(1, 2)))

        z_pooled = self.graph_pool(readout, G.mean(dim=-3))
        z_final = torch.cat([self.node_pool(z_pooled), self.node_readout(torch.cat(mus, dim=-1)).flatten(1)], dim=-1)
        return {
            "logits": self.classifier(z_final),
            "A": A,
            "z": zs,
            "mu": mus,
            "logvar": logvars,
            "A_recon": A_recon,
            "z_contrast": self.contrast_proj(z_pooled),
        }


def build_model(cfg):
    mc, dc = cfg["model"], cfg["data"]
    return DynaGraph(
        num_features=mc["num_features"],
        num_classes=mc["num_classes"],
        num_groups=dc["n_blocks"],
        lstm_hidden=mc["lstm_hidden"],
        lstm_layers=mc["lstm_layers"],
        gin_dims=mc["gin_dims"],
        latent_dim=mc["latent_dim"],
        k_neighbors=mc["k_neighbors"],
        graph_rank=mc["graph_rank"],
        dropout=mc["dropout"],
        pool_clusters=mc["pool_clusters"],
        node_readout_dim=mc["node_readout_dim"],
        alpha_init=mc["alpha_init"],
    )
