"""
MFMN core model -- factorized latent space with target-specific pathways.
Trimmed to the classes actually used by this package's 5 benchmark notebooks:
MFMN (base), MFMN_Interact (adds FM-style factor interactions),
MFMN_InteractAux (adds per-factor auxiliary supervision -- used for VDss),
MFMN_DynGate (adds input-conditioned GLU gating on top of InteractAux --
used for CL and Fu). Ablation-only variants (MFMN_Gated, MFMN_BioInteract,
MFMN_Residual, MFMN_InteractMask, MFMN_Attention, evidential variants) are
omitted -- they were tried, found not to improve on DynGate/InteractAux for
these targets, and are not part of any of the 5 final recipes reproduced
here. See the full mfmn/deliverables/PROJECT_REPORT.md for that history.
"""
import torch
import torch.nn as nn

FACTOR_DIM = 8  # small per Section 15 (bottleneck, not huge hidden dims)


class MFMN(nn.Module):
    def __init__(self, context_dim, factor_raw_dims, factor_names, targets=("CL", "VDss"),
                 factor_dim=FACTOR_DIM, dropout=0.2):
        super().__init__()
        self.factor_names = list(factor_names)
        self.targets = list(targets)
        self.factor_proj = nn.ModuleDict({
            f: nn.Sequential(
                nn.Linear(context_dim + factor_raw_dims[f], factor_dim), nn.SiLU(), nn.Dropout(dropout))
            for f in self.factor_names
        })
        latent_dim = factor_dim * len(self.factor_names)
        self.pathways = nn.ModuleDict({
            t: nn.Sequential(
                nn.Linear(latent_dim, 16), nn.SiLU(), nn.Dropout(dropout), nn.Linear(16, 1))
            for t in self.targets
        })

    def forward(self, context, factor_raw, return_factors=False):
        zs = []
        for f in self.factor_names:
            inp = torch.cat([context, factor_raw[f]], dim=1)
            zs.append(self.factor_proj[f](inp))
        latent = torch.cat(zs, dim=1)
        out = {t: self.pathways[t](latent).squeeze(-1) for t in self.targets}
        if return_factors:
            return out, zs
        return out


def count_params(model):
    return sum(p.numel() for p in model.parameters())


class MFMN_Interact(nn.Module):
    """
    Experiment E: adds an explicit, low-rank factor-interaction term on top
    of the B+D architecture (Section 12).

    Mechanism (Factorization-Machine style, Rendle 2010): rather than a full
    O(F^2) pairwise interaction tensor (which would badly overfit ~1,300
    compounds), each factor is projected to a small rank-k interaction
    embedding v_i, and all pairwise dot products <v_i, v_j> are computed in
    closed form via the standard FM identity:

        sum_{i<j} v_i . v_j  =  0.5 * [ (sum_i v_i)^2 - sum_i v_i^2 ]   (elementwise over k)

    computed per-target (separate v_i^CL vs v_i^VDss projections), so CL and
    VDss can learn different interaction structures from the same 7 factors
    -- directly implementing Section 11's "do not force CL/VDss to share
    factor importance" together with Section 12's interaction requirement.

    The interaction vector is concatenated with (not replacing) the plain
    factor concatenation from B+D, so the model can fall back toward the
    B+D solution if interactions don't help for a given target -- the
    interaction term is additive capacity, not a forced replacement.
    """
    def __init__(self, context_dim, factor_raw_dims, factor_names, targets=("CL", "VDss"),
                 factor_dim=FACTOR_DIM, interact_rank=8, dropout=0.2):
        super().__init__()
        self.factor_names = list(factor_names)
        self.targets = list(targets)
        self.factor_proj = nn.ModuleDict({
            f: nn.Sequential(
                nn.Linear(context_dim + factor_raw_dims[f], factor_dim), nn.SiLU(), nn.Dropout(dropout))
            for f in self.factor_names
        })
        # per-target, per-factor interaction-rank projections
        self.interact_proj = nn.ModuleDict({
            t: nn.ModuleDict({f: nn.Linear(factor_dim, interact_rank) for f in self.factor_names})
            for t in self.targets
        })
        latent_dim = factor_dim * len(self.factor_names) + interact_rank
        self.pathways = nn.ModuleDict({
            t: nn.Sequential(
                nn.Linear(latent_dim, 16), nn.SiLU(), nn.Dropout(dropout), nn.Linear(16, 1))
            for t in self.targets
        })

    def forward(self, context, factor_raw, return_factors=False):
        zs = []
        for f in self.factor_names:
            inp = torch.cat([context, factor_raw[f]], dim=1)
            zs.append(self.factor_proj[f](inp))
        concat_z = torch.cat(zs, dim=1)

        out = {}
        for t in self.targets:
            vs = torch.stack([self.interact_proj[t][f](z) for f, z in zip(self.factor_names, zs)], dim=1)
            # vs: (batch, n_factors, interact_rank)
            sum_sq = vs.sum(dim=1) ** 2
            sq_sum = (vs ** 2).sum(dim=1)
            interaction = 0.5 * (sum_sq - sq_sum)  # (batch, interact_rank)
            latent_t = torch.cat([concat_z, interaction], dim=1)
            out[t] = self.pathways[t](latent_t).squeeze(-1)
        if return_factors:
            return out, zs
        return out


class MFMN_InteractAux(MFMN_Interact):
    """
    Experiment C+E combined: adds a small auxiliary-supervision head per
    factor on top of MFMN_Interact (Section 10). Each head predicts one
    real, legitimate molecular property matching that factor's mechanism
    (e.g. Z_lipophilicity -> predicted MolLogP), regularizing the factor to
    faithfully encode what it is supposed to represent rather than let the
    downstream interaction/pathway heads treat it as an arbitrary embedding.

    Auxiliary loss is used only during training (added to the total loss,
    weighted by aux_weight); the auxiliary heads are not used at inference.
    """
    def __init__(self, context_dim, factor_raw_dims, factor_names, targets=("CL", "VDss"),
                 factor_dim=8, interact_rank=8, dropout=0.2):
        super().__init__(context_dim, factor_raw_dims, factor_names, targets, factor_dim, interact_rank, dropout)
        self.aux_heads = nn.ModuleDict({f: nn.Linear(factor_dim, 1) for f in self.factor_names})

    def forward_with_aux(self, context, factor_raw):
        zs = []
        for f in self.factor_names:
            inp = torch.cat([context, factor_raw[f]], dim=1)
            zs.append(self.factor_proj[f](inp))
        concat_z = torch.cat(zs, dim=1)

        aux_preds = {f: self.aux_heads[f](z).squeeze(-1) for f, z in zip(self.factor_names, zs)}

        out = {}
        for t in self.targets:
            vs = torch.stack([self.interact_proj[t][f](z) for f, z in zip(self.factor_names, zs)], dim=1)
            sum_sq = vs.sum(dim=1) ** 2
            sq_sum = (vs ** 2).sum(dim=1)
            interaction = 0.5 * (sum_sq - sq_sum)
            latent_t = torch.cat([concat_z, interaction], dim=1)
            out[t] = self.pathways[t](latent_t).squeeze(-1)
        return out, aux_preds


class MFMN_DynGate(MFMN_InteractAux):
    """
    Dynamic (input-conditioned) gating -- the key difference from
    MFMN_Gated (Group C): that gate was a single learned scalar per
    (factor, target), IDENTICAL for every molecule. This gate is a
    function of the molecule's own context+raw descriptors, so different
    compounds can rely on a factor to different degrees -- the
    "attention-adjacent" idea (per-compound, data-dependent weighting)
    without the parameter/overfitting risk of full multi-head attention.

    Mechanism: a GLU (Gated Linear Unit) replaces the plain Linear+SiLU
    factor projection --
        value = SiLU(Linear_value(inp))
        gate  = sigmoid(Linear_gate(inp))
        z_f   = value * gate
    where inp = concat([context, factor_raw[f]]), same input as the
    ungated version. Doubles the factor-projection parameter count (two
    Linear layers instead of one) but changes nothing else -- same
    interaction term, same auxiliary heads, same pathways -- so any
    change in accuracy is attributable to the gating mechanism alone.
    """
    def __init__(self, context_dim, factor_raw_dims, factor_names, targets=("CL", "VDss"),
                 factor_dim=8, interact_rank=8, dropout=0.2):
        super().__init__(context_dim, factor_raw_dims, factor_names, targets, factor_dim, interact_rank, dropout)
        del self.factor_proj  # unused here -- replaced by the value/gate branches below
        self.factor_value = nn.ModuleDict({
            f: nn.Sequential(
                nn.Linear(context_dim + factor_raw_dims[f], factor_dim), nn.SiLU(), nn.Dropout(dropout))
            for f in self.factor_names
        })
        self.factor_gate = nn.ModuleDict({
            f: nn.Sequential(nn.Linear(context_dim + factor_raw_dims[f], factor_dim), nn.Sigmoid())
            for f in self.factor_names
        })

    def forward_with_aux(self, context, factor_raw):
        zs = []
        gates = {}
        for f in self.factor_names:
            inp = torch.cat([context, factor_raw[f]], dim=1)
            g = self.factor_gate[f](inp)
            gates[f] = g
            zs.append(self.factor_value[f](inp) * g)
        concat_z = torch.cat(zs, dim=1)
        aux_preds = {f: self.aux_heads[f](z).squeeze(-1) for f, z in zip(self.factor_names, zs)}

        out = {}
        for t in self.targets:
            vs = torch.stack([self.interact_proj[t][f](z) for f, z in zip(self.factor_names, zs)], dim=1)
            sum_sq = vs.sum(dim=1) ** 2
            sq_sum = (vs ** 2).sum(dim=1)
            interaction = 0.5 * (sum_sq - sq_sum)
            latent_t = torch.cat([concat_z, interaction], dim=1)
            out[t] = self.pathways[t](latent_t).squeeze(-1)
        self._last_gates = {f: gates[f].detach().cpu().numpy() for f in self.factor_names}
        return out, aux_preds

    def forward(self, context, factor_raw, return_factors=False):
        out, _ = self.forward_with_aux(context, factor_raw)
        return out

