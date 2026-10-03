"""
MFMN-v3, E13: dense-adjacency message-passing GNN.

Architecture directly informed by the pKa-prediction GNN literature reviewed
for this experiment (QupKake, JCTC 2024; GR-pKa, Bioinformatics 2024 -- see
mfmn_v3/V3_FINDINGS.md): xTB features injected as node/edge inputs BEFORE
message passing (not bolted on afterward, which is what E11/E12 did and
which both failed), a per-atom message-passing update, and a final readout
that combines a specific ionization-site atom's embedding with a
whole-molecule pooled embedding -- exactly the site-plus-global combination
those papers use, but here with a fully LEARNED local representation instead
of a handful of fixed QM scalars.

Implementation notes: dense/padded batching (no torch_geometric/dgl -- see
graph_features.py docstring for why), edge-conditioned message passing
(each edge's message is gated by its own feature vector, similar in spirit
to D-MPNN / edge-conditioned convolution), residual connections between
layers, masked mean+max pooling for the global readout.
"""
import torch
import torch.nn as nn


class MPNNLayer(nn.Module):
    """One round of message passing: a SHARED linear transform of each
    neighbor's state, gated by a small edge-feature-conditioned vector, then
    a GRU-style update combines the aggregated message with the node's
    previous state.

    Deliberately NOT full edge-conditioned convolution (a distinct H x H
    transform matrix per edge, as in Simonovsky & Komodakis 2017): at this
    project's molecule sizes (heavy-atom counts up to ~90 in the pKa_Acidic
    pool) and a hidden_dim of 128, a full per-edge matrix would need
    O(B * N^2 * H^2) memory -- tens of GB for a normal batch size on this
    box's 20GB MIG partition. The gated design keeps the O(N^2) term scaling
    with H, not H^2, while still letting the xTB Wiberg bond order and bond
    type modulate how strongly information flows across each edge."""

    def __init__(self, hidden_dim, edge_dim):
        super().__init__()
        self.msg_linear = nn.Linear(hidden_dim, hidden_dim)
        self.edge_gate = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid(),
        )
        self.update = nn.GRUCell(hidden_dim, hidden_dim)
        self.hidden_dim = hidden_dim

    def forward(self, h, adj_feats, adj_mask):
        # h: [B,N,H]  adj_feats: [B,N,N,E]  adj_mask: [B,N,N]
        B, N, H = h.shape
        h_transformed = self.msg_linear(h)                 # [B,N,H], cost O(B*N*H^2)
        gate = self.edge_gate(adj_feats)                    # [B,N(i),N(j),H], cost O(B*N^2*edge_dim*H)
        h_j = h_transformed.unsqueeze(1).expand(B, N, N, H)  # broadcast neighbor state over the i axis
        msg = gate * h_j * adj_mask.unsqueeze(-1)
        agg = msg.sum(dim=2)                                 # sum over neighbors j -> [B,N,H]
        h_flat = h.reshape(B * N, H)
        agg_flat = agg.reshape(B * N, H)
        h_new = self.update(agg_flat, h_flat).view(B, N, H)
        return h_new


class GraphMPNN(nn.Module):
    """Full model: input projection -> K message-passing layers (residual) ->
    site-atom + global-pooled readout -> task heads.

    `pretrain_heads`: dict[name -> out_dim] for graph-level self-supervised
    targets (whole-molecule xTB scalars). `node_pretrain_dim`: if >0, adds a
    per-node head predicting that many node-level targets (here: the atom's
    own xTB charge) -- trained on EVERY atom, not just the site, so the model
    learns a generally-informative local representation, not one narrowly
    tuned to acidic sites only.

    `site_readout`: if False, the readout is [mean-pool, max-pool] only (no
    ionizable-site embedding); used for the final pKa_Basic model. The node-level
    site flag in the input features is unaffected.
    """

    def __init__(self, node_dim, edge_dim, hidden_dim=128, n_layers=4, dropout=0.1,
                 pretrain_heads=None, node_pretrain_dim=0, site_readout=True):
        super().__init__()
        self.input_proj = nn.Sequential(nn.Linear(node_dim, hidden_dim), nn.SiLU())
        self.layers = nn.ModuleList([MPNNLayer(hidden_dim, edge_dim) for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout)
        self.hidden_dim = hidden_dim
        self.site_readout = site_readout

        readout_dim = hidden_dim * (3 if site_readout else 2)  # [site +] mean-pool + max-pool
        self.readout_dim = readout_dim
        self.readout_norm = nn.LayerNorm(readout_dim)

        self.pretrain_heads = nn.ModuleDict()
        for name, out_dim in (pretrain_heads or {}).items():
            self.pretrain_heads[name] = nn.Sequential(
                nn.Linear(readout_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, out_dim))

        self.node_pretrain_head = None
        if node_pretrain_dim > 0:
            self.node_pretrain_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, node_pretrain_dim))

        self.finetune_head = None  # set via add_finetune_head once pretraining is done

    def add_finetune_head(self, out_dim=1, hidden=None):
        hidden = hidden or self.hidden_dim
        self.finetune_head = nn.Sequential(
            nn.Linear(self.readout_dim, hidden), nn.SiLU(), nn.Dropout(0.2), nn.Linear(hidden, out_dim))
        return self

    def encode(self, nodes, adj_feats, adj_mask, node_mask, site_idx):
        h = self.input_proj(nodes)
        for layer in self.layers:
            h_new = layer(h, adj_feats, adj_mask)
            h = h + self.dropout(h_new)
            h = h * node_mask.unsqueeze(-1)  # re-zero padding after residual add

        # masked mean/max pool over valid atoms
        mask = node_mask.unsqueeze(-1)                     # [B,N,1]
        mean_pool = (h * mask).sum(1) / mask.sum(1).clamp(min=1.0)
        h_masked_for_max = h.masked_fill(mask == 0, float("-inf"))
        max_pool, _ = h_masked_for_max.max(dim=1)
        max_pool = torch.where(torch.isfinite(max_pool), max_pool, torch.zeros_like(max_pool))

        if not self.site_readout:
            return h, self.readout_norm(torch.cat([mean_pool, max_pool], dim=-1))

        B, N, H = h.shape
        has_site = (site_idx >= 0)
        safe_idx = site_idx.clamp(min=0)
        site_embed = h[torch.arange(B, device=h.device), safe_idx]
        site_embed = site_embed * has_site.float().unsqueeze(-1)  # zero out when no site detected

        readout = torch.cat([site_embed, mean_pool, max_pool], dim=-1)
        readout = self.readout_norm(readout)
        return h, readout

    def forward_pretrain(self, nodes, adj_feats, adj_mask, node_mask, site_idx):
        h, readout = self.encode(nodes, adj_feats, adj_mask, node_mask, site_idx)
        graph_preds = {name: head(readout) for name, head in self.pretrain_heads.items()}
        node_preds = self.node_pretrain_head(h) if self.node_pretrain_head is not None else None
        return graph_preds, node_preds

    def forward_finetune(self, nodes, adj_feats, adj_mask, node_mask, site_idx):
        _, readout = self.encode(nodes, adj_feats, adj_mask, node_mask, site_idx)
        return self.finetune_head(readout).squeeze(-1)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
