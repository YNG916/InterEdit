import torch
import torch.nn as nn
from models.freq_utils import dct_ii, band_energy, default_bands
from models.utils import *
from models.cfg_sampler import ClassifierFreeSampleModel
# from models.cfg_sampler_seperate import ClassifierFreeSampleModel
from models.blocks import *
from utils.utils import *
import torch.nn.functional as F
from models.interclip_teacher import build_interclip_teacher, InterCLIPTeacherWrapper

from models.gaussian_diffusion import (
    MotionDiffusion,
    space_timesteps,
    get_named_beta_schedule,
    create_named_schedule_sampler,
    ModelMeanType,
    ModelVarType,
    LossType
)

import torch.distributed as dist

def _reduce_over_k(loss_bk: torch.Tensor, mode: str = "mean") -> torch.Tensor:
    """
    Reduce token-wise loss over K plan tokens.
    loss_bk: (B,K) -> (B,)
    """
    mode = str(mode).lower()
    if mode == "mean":
        return loss_bk.mean(dim=1)
    if mode == "min":
        return loss_bk.min(dim=1).values
    if mode == "max":
        return loss_bk.max(dim=1).values
    raise ValueError(f"Unknown PLAN_REDUCE_K={mode}")


def _dist_is_on() -> bool:
    return dist.is_available() and dist.is_initialized()


@torch.no_grad()
def _all_gather_cat(x: torch.Tensor) -> torch.Tensor:
    if not _dist_is_on():
        return x
    world = dist.get_world_size()
    xs = [torch.zeros_like(x) for _ in range(world)]
    dist.all_gather(xs, x.contiguous())
    return torch.cat(xs, dim=0)


class MotionEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.input_feats = cfg.INPUT_DIM
        self.latent_dim = cfg.LATENT_DIM
        self.ff_size = cfg.FF_SIZE
        self.num_layers = cfg.NUM_LAYERS
        self.num_heads = cfg.NUM_HEADS
        self.dropout = cfg.DROPOUT
        self.activation = cfg.ACTIVATION

        self.query_token = nn.Parameter(torch.randn(1, self.latent_dim))

        self.embed_motion = nn.Linear(self.input_feats * 2, self.latent_dim)
        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout, max_len=2000)

        seqTransEncoderLayer = nn.TransformerEncoderLayer(
            d_model=self.latent_dim,
            nhead=self.num_heads,
            dim_feedforward=self.ff_size,
            dropout=self.dropout,
            activation=self.activation,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(seqTransEncoderLayer, num_layers=self.num_layers)
        self.out_ln = nn.LayerNorm(self.latent_dim)
        self.out = nn.Linear(self.latent_dim, 512)

    def forward(self, batch):
        x, mask = batch["motions"], batch["mask"]
        B, T, D = x.shape

        x = x.reshape(B, T, 2, -1)[..., :-4].reshape(B, T, -1)

        x_emb = self.embed_motion(x)

        emb = torch.cat([self.query_token[torch.zeros(B, dtype=torch.long, device=x.device)][:, None], x_emb], dim=1)

        seq_mask = (mask > 0.5)
        token_mask = torch.ones((B, 1), dtype=bool, device=x.device)
        valid_mask = torch.cat([token_mask, seq_mask], dim=1)

        h = self.sequence_pos_encoder(emb)
        h = self.transformer(h, src_key_padding_mask=~valid_mask)
        h = self.out_ln(h)
        motion_emb = self.out(h[:, 0])

        batch["motion_emb"] = motion_emb
        return batch


class SourceMotionEncoder(nn.Module):
    """
    Encode source motions (B,T,2*INPUT_DIM) into a compact embedding (B,512).
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.input_feats = cfg.INPUT_DIM
        self.latent_dim = cfg.LATENT_DIM
        self.ff_size = cfg.FF_SIZE
        self.num_layers = cfg.NUM_LAYERS
        self.num_heads = cfg.NUM_HEADS
        self.dropout = cfg.DROPOUT
        self.activation = cfg.ACTIVATION

        self.query_token = nn.Parameter(torch.randn(1, self.latent_dim))

        self.embed_motion = nn.Linear((self.input_feats - 4) * 2, self.latent_dim)
        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout, max_len=2000)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.latent_dim,
            nhead=self.num_heads,
            dim_feedforward=self.ff_size,
            dropout=self.dropout,
            activation=self.activation,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=self.num_layers)
        self.out_ln = nn.LayerNorm(self.latent_dim)
        self.out = nn.Linear(self.latent_dim, 512)

    def forward(self, motions, mask):
        """
        motions: (B,T,2*INPUT_DIM)
        mask:   (B,T)  1=valid 0=pad
        """
        B, T, D = motions.shape
        x = motions.reshape(B, T, 2, -1)[..., :-4].reshape(B, T, -1)  # (B,T,(INPUT_DIM-4)*2)

        x_emb = self.embed_motion(x)  # (B,T,latent)
        q = self.query_token[torch.zeros(B, dtype=torch.long, device=motions.device)][:, None]
        emb = torch.cat([q, x_emb], dim=1)  # (B,T+1,latent)

        token_mask = torch.ones((B, 1), dtype=torch.bool, device=motions.device)
        valid_mask = torch.cat([token_mask, (mask > 0.5)], dim=1)  # (B,T+1)

        h = self.sequence_pos_encoder(emb)
        h = self.transformer(h, src_key_padding_mask=~valid_mask)
        h = self.out_ln(h)
        source_emb = self.out(h[:, 0])  # (B,512)
        return source_emb


class InterEditDenoiser(nn.Module):
    def __init__(self,
                 input_feats,
                 latent_dim=512,
                 num_frames=240,
                 ff_size=1024,
                 num_layers=8,
                 num_heads=8,
                 dropout=0.1,
                 activation="gelu",
                 cfg_weight=0.,
                 **kargs):
        super().__init__()
        self.cfg = kargs['cfg']
        self.cfg_weight = cfg_weight
        self.num_frames = num_frames
        self.latent_dim = latent_dim
        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.activation = activation
        self.input_feats = input_feats
        self.time_embed_dim = latent_dim

        self.text_emb_dim = 768
        self.source_emb_dim = 512

        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, dropout=0)
        self.embed_timestep = TimestepEmbedder(self.latent_dim, self.sequence_pos_encoder)

        # Input Embedding
        self.motion_embed = nn.Linear(self.input_feats, self.latent_dim)
        self.text_embed = nn.Linear(self.text_emb_dim, self.latent_dim)
        self.source_embed = nn.Linear(self.source_emb_dim, self.latent_dim)

        self.blocks = nn.ModuleList()
        dpr = [x.item() for x in torch.linspace(0, dropout, num_layers + 1)]
        for i in range(num_layers):
            self.blocks.append(
                InterEditTransformerBlock(
                    num_heads=num_heads,
                    latent_dim=latent_dim,
                    dropout=dpr[i],
                    ff_size=ff_size,
                    num_layers=num_layers,
                    cur_layer=i,
                    LPA=kargs['cfg'].get('LPA', False),
                    cfg=kargs['cfg']
                )
            )
        
        # --- plan tokens ---
        self.num_plan = kargs['cfg'].get('NUM_PLAN', 16)
        self.use_plan = bool(kargs['cfg'].get('USE_PLAN', True))
        if self.use_plan:
            self.plan_tokens = nn.Parameter(torch.randn(1, self.num_plan, self.latent_dim * 2) * 0.02)
            self.plan_proj_token = nn.Sequential(
                nn.LayerNorm(self.latent_dim * 2),
                nn.Linear(self.latent_dim * 2, 512)
            )
        else:
            self.plan_tokens = None
            self.plan_proj_token = None


        # which layer to tap plan for loss
        self.plan_layer_idx = int(kargs['cfg'].get('PLAN_LAYER_IDX', self.num_layers // 2))

        self._plan_pred_tokens = None  # (B,K,512)

        # --- freq tokens (S/D, 3 bands each => 6 tokens) ---
        self.use_freq = bool(kargs['cfg'].get('USE_FREQ', True))
        self.freq_drop_prob = float(kargs['cfg'].get('FREQ_DROP_PROB', 0.0))

        self.freq_feat_dim = int(kargs['cfg'].get('FREQ_FEAT_DIM', self.input_feats - 4))

        self.num_freq = 6  # S_low,S_mid,S_high,D_low,D_mid,D_high

        self.freq_token_proj = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(self.freq_feat_dim),
                nn.Linear(self.freq_feat_dim, self.latent_dim * 2)
            ) for _ in range(self.num_freq)
        ])

        self.freq_head = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(self.latent_dim * 2),
                nn.Linear(self.latent_dim * 2, self.freq_feat_dim)
            ) for _ in range(self.num_freq)
        ])

        self._freq_pred = None
        self.freq_layer_idx = int(kargs['cfg'].get('FREQ_LAYER_IDX', self.num_layers // 2))

        self.out = zero_module(FinalLayer(self.latent_dim, self.input_feats))

    def forward(self, x, timesteps, mask=None, cond=None, source_emb=None):
        """
        x: (B,T,2*D)
        cond: (B,768)
        source_emb: (B,512)
        """
        B, T = x.shape[0], x.shape[1]
        x_a, x_b = x[..., :self.input_feats], x[..., self.input_feats:]

        if mask is not None:
            mask = mask[..., 0]  # (B,T)

        emb = self.embed_timestep(timesteps) + self.text_embed(cond)
        if source_emb is not None:
            emb = emb + self.source_embed(source_emb)

        a_emb = self.motion_embed(x_a)
        b_emb = self.motion_embed(x_b)
        h_a_prev = self.sequence_pos_encoder(a_emb)
        h_b_prev = self.sequence_pos_encoder(b_emb)

        if mask is None:
            mask = torch.ones(B, T, device=x_a.device)
        key_padding_mask = ~(mask > 0.5)
        
        plan = None
        if self.use_plan and (self.plan_tokens is not None):
            plan = self.plan_tokens.expand(B, -1, -1)  # (B,Kp,2C)

        freq = None
        self._freq_pred = None

        if self.use_freq:
            drop = (self.training and (self.freq_drop_prob > 0.0) and (torch.rand(1, device=x.device) < self.freq_drop_prob))
            if not drop:
                xa = x_a[..., :self.freq_feat_dim]
                xb = x_b[..., :self.freq_feat_dim]
                S = 0.5 * (xa + xb)
                D = (xa - xb)

                if mask is None:
                    m = torch.ones((B, T), device=x.device, dtype=xa.dtype)
                else:
                    m = mask.to(dtype=xa.dtype)  # (B,T)
                S = S * m[:, :, None]
                D = D * m[:, :, None]

                # DCT along time
                Cs = dct_ii(S, dim=1, norm="ortho")  # (B,T,Dfeat)
                Cd = dct_ii(D, dim=1, norm="ortho")

                (l0,l1),(m0,m1),(h0,h1) = default_bands(T,
                                                       float(self.cfg.get('FREQ_R_LOW', 0.08)),
                                                       float(self.cfg.get('FREQ_R_MID', 0.25)),
                                                       float(self.cfg.get('FREQ_R_HI', 0.50)))

                Fs_low  = band_energy(Cs, l0, l1)  # (B,Dfeat)
                Fs_mid  = band_energy(Cs, m0, m1)
                Fs_high = band_energy(Cs, h0, h1)
                Fd_low  = band_energy(Cd, l0, l1)
                Fd_mid  = band_energy(Cd, m0, m1)
                Fd_high = band_energy(Cd, h0, h1)

                feats = [Fs_low, Fs_mid, Fs_high, Fd_low, Fd_mid, Fd_high]
                freq = torch.stack([self.freq_token_proj[i](feats[i]) for i in range(6)], dim=1)  # (B,6,2C)

        self._plan_pred_tokens = None
        self._freq_pred = None

        for li, block in enumerate(self.blocks):
            h_a, h_b, plan, freq = block(h_a_prev, h_b_prev, plan, freq=freq, emb=emb, key_padding_mask=key_padding_mask)
            h_a_prev, h_b_prev = h_a, h_b

            # tap plan tokens for plan loss
            if self.use_plan and (plan is not None) and (li == self.plan_layer_idx):
                self._plan_pred_tokens = self.plan_proj_token(plan)

            # tap freq tokens for freq regression loss
            if self.use_freq and (freq is not None) and (li == self.freq_layer_idx):
                # (B,6,Dfeat)
                preds = []
                for i in range(6):
                    preds.append(self.freq_head[i](freq[:, i, :]))
                self._freq_pred = torch.stack(preds, dim=1)


        output_a = self.out(h_a)
        output_b = self.out(h_b)
        output = torch.cat([output_a, output_b], dim=-1)

        if self.use_plan and (plan is not None) and (self._plan_pred_tokens is None):
            self._plan_pred_tokens = self.plan_proj_token(plan)

        if self.use_freq and (freq is not None) and (self._freq_pred is None):
            preds = []
            for i in range(6):
                preds.append(self.freq_head[i](freq[:, i, :]))
            self._freq_pred = torch.stack(preds, dim=1)

        return output


class InterEditDiffusion(nn.Module):
    def __init__(self, cfg, sampling_strategy="ddim50"):
        super().__init__()
        self.cfg = cfg
        self.nfeats = cfg.INPUT_DIM
        self.latent_dim = cfg.LATENT_DIM
        self.ff_size = cfg.FF_SIZE
        self.num_layers = cfg.NUM_LAYERS
        self.num_heads = cfg.NUM_HEADS
        self.dropout = cfg.DROPOUT
        self.activation = cfg.ACTIVATION
        self.motion_rep = cfg.MOTION_REP

        self.cfg_weight = cfg.CFG_WEIGHT
        self.diffusion_steps = cfg.DIFFUSION_STEPS
        self.beta_scheduler = cfg.BETA_SCHEDULER
        self.sampler = cfg.SAMPLER
        self.sampling_strategy = sampling_strategy

        self.net = InterEditDenoiser(
            self.nfeats,
            self.latent_dim,
            ff_size=self.ff_size,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            dropout=self.dropout,
            activation=self.activation,
            cfg_weight=self.cfg_weight,
            cfg=cfg
        )

        # source encoder
        self.motion_encoder = SourceMotionEncoder(cfg)

        # --- Motion teacher for plan loss ---
        self.plan_loss_w = float(getattr(cfg, "PLAN_LOSS_W", 0.05))
        self.use_plan = bool(getattr(cfg, "USE_PLAN", True))
        self.plan_loss_type = str(getattr(cfg, "PLAN_LOSS_TYPE", "infonce")).lower()
        self.plan_cos_w = float(getattr(cfg, "PLAN_COS_W", 1.0))
        self.plan_mse_w = float(getattr(cfg, "PLAN_MSE_W", 1.0))
        self.plan_reduce_k = str(getattr(cfg, "PLAN_REDUCE_K", "mean")).lower()
        teacher_cfg_path = getattr(cfg, "TEACHER_CFG", "configs/eval_model.yaml")
        teacher_ckpt = getattr(cfg, "TEACHER_CKPT", "eval_model/interclip.ckpt")

        teacher = build_interclip_teacher(teacher_cfg_path, ckpt_path=teacher_ckpt)
        self.teacher = InterCLIPTeacherWrapper(teacher)

        # --- freq loss ---
        self.use_freq = bool(getattr(cfg, "USE_FREQ", True))
        self.freq_loss_w = float(getattr(cfg, "FREQ_LOSS_W", 0.01))
        self.freq_high_w = float(getattr(cfg, "FREQ_HIGH_W", 0.25))
        self.freq_feat_dim = int(getattr(cfg, "FREQ_FEAT_DIM", self.nfeats - 4))
        self.freq_r_low = float(getattr(cfg, "FREQ_R_LOW", 0.08))
        self.freq_r_mid = float(getattr(cfg, "FREQ_R_MID", 0.25))
        self.freq_r_hi  = float(getattr(cfg, "FREQ_R_HI", 0.50))


        self.betas = get_named_beta_schedule(self.beta_scheduler, self.diffusion_steps)

        timestep_respacing = [self.diffusion_steps]
        self.diffusion = MotionDiffusion(
            use_timesteps=space_timesteps(self.diffusion_steps, timestep_respacing),
            betas=self.betas,
            motion_rep=self.motion_rep,
            model_mean_type=ModelMeanType.START_X,
            model_var_type=ModelVarType.FIXED_SMALL,
            loss_type=LossType.MSE,
            rescale_timesteps=False,
        )
        self.sampler = create_named_schedule_sampler(self.sampler, self.diffusion)

    def mask_cond(self, cond, cond_mask_prob=0.1, force_mask=False):
        bs = cond.shape[0]
        if force_mask:
            return torch.zeros_like(cond), torch.zeros((bs, 1), device=cond.device)
        elif cond_mask_prob > 0.:
            mask = torch.bernoulli(torch.ones(bs, device=cond.device) * cond_mask_prob).view([bs] + [1] * (len(cond.shape) - 1))
            # return masked_cond, keep_mask(1=keep,0=drop)
            keep = (1. - mask)
            return cond * keep, keep
        else:
            return cond, None

    def generate_src_mask(self, T, length):
        B = length.shape[0]
        src_mask = torch.ones(B, T, 2)
        for p in range(2):
            for i in range(B):
                for j in range(length[i], T):
                    src_mask[i, j, p] = 0
        return src_mask

    def compute_loss(self, batch):
        cond = batch["cond"]
        x_start = batch["motions"]       # target
        sources = batch["sources"]       # source
        B, T = x_start.shape[:2]

        cond, keep_mask = self.mask_cond(cond, 0.1)  # keep_mask: (B,1) or None

        # source mask -> source embedding
        src_mask = self.generate_src_mask(T, batch["source_lens"]).to(x_start.device)  # (B,T,2)
        src_mask_1 = src_mask[..., 0]  # (B,T)
        source_emb = self.motion_encoder(sources, src_mask_1)  # (B,512)

        if keep_mask is not None:
            # keep_mask: 1=keep,0=drop
            source_emb = source_emb * keep_mask.view(B, 1)

        # target seq mask
        tgt_mask = self.generate_src_mask(T, batch["motion_lens"]).to(x_start.device)

        t, _ = self.sampler.sample(B, x_start.device)
        output = self.diffusion.training_losses(
            model=self.net,
            x_start=x_start,
            t=t,
            mask=tgt_mask,
            t_bar=self.cfg.T_BAR,
            cond_mask=keep_mask,
            model_kwargs={
                "mask": tgt_mask,
                "cond": cond,
                "source_emb": source_emb,
            },
        )


        # --- plan loss: token-wise InfoNCE with GLOBAL in-batch negatives (DDP all_gather) ---
        # --- types (infonce/cos/mse/combos), token-level then reduce over K ---
        if self.use_plan and (self.plan_loss_w > 0):
            tau = float(getattr(self.cfg, "PLAN_TAU", 0.1))

            # teacher target embedding
            with torch.no_grad():
                E_tgt = self.teacher.motion_emb(x_start, batch["motion_lens"])  # (B,512)
                E_tgt_n = F.normalize(E_tgt, dim=-1)                            # (B,512)

            pred_tokens = self.net._plan_pred_tokens  # (B,K,512) or None
            if pred_tokens is None:
                if (self.net.plan_tokens is None) or (self.net.plan_proj_token is None):
                    plan_loss = output["total"].sum() * 0.0
                    output["plan_loss"] = plan_loss
                    pred_tokens = None
                else:
                    pred_tokens = self.net.plan_proj_token(self.net.plan_tokens.expand(B, -1, -1))  # (B,K,512)

            if pred_tokens is not None:
                keep = (keep_mask.view(-1) > 0.5) if (keep_mask is not None) else None

                def _apply_keep(loss_b: torch.Tensor) -> torch.Tensor:
                    if keep is None:
                        return loss_b.mean()
                    if keep.any():
                        return loss_b[keep].mean()
                    return loss_b.sum() * 0.0

                def compute_cos() -> torch.Tensor:
                    pred_n = F.normalize(pred_tokens, dim=-1)     # (B,K,512)
                    tgt_n  = E_tgt_n[:, None, :]                  # (B,1,512)
                    cos_bk = (pred_n * tgt_n).sum(dim=-1)         # (B,K)
                    loss_bk = 1.0 - cos_bk                        # (B,K)
                    loss_b  = _reduce_over_k(loss_bk, self.plan_reduce_k)  # (B,)
                    return _apply_keep(loss_b)

                def compute_mse() -> torch.Tensor:
                    diff = pred_tokens - E_tgt[:, None, :]        # (B,K,512)
                    loss_bk = (diff * diff).mean(dim=-1)          # (B,K)
                    loss_b  = _reduce_over_k(loss_bk, self.plan_reduce_k)  # (B,)
                    return _apply_keep(loss_b)

                def compute_infonce() -> torch.Tensor:
                    E_all = _all_gather_cat(E_tgt_n)              # (B_global,512)
                    pred_n = F.normalize(pred_tokens, dim=-1)     # (B,K,512)

                    if _dist_is_on():
                        rank = dist.get_rank()
                        B_local = E_tgt_n.shape[0]
                        labels = torch.arange(B_local, device=pred_n.device) + rank * B_local
                    else:
                        labels = torch.arange(E_tgt_n.shape[0], device=pred_n.device)

                    logits = torch.einsum("bkd,nd->bkn", pred_n, E_all) / tau  # (B,K,Bg)
                    B_local, K, B_global = logits.shape
                    logits_flat = logits.reshape(B_local * K, B_global)
                    labels_flat = labels[:, None].expand(B_local, K).reshape(-1)

                    if keep is not None:
                        keep_flat = keep[:, None].expand(B_local, K).reshape(-1)
                        if keep_flat.any():
                            return F.cross_entropy(logits_flat[keep_flat], labels_flat[keep_flat])
                        return logits_flat.sum() * 0.0
                    return F.cross_entropy(logits_flat, labels_flat)

                # ---- choose ----
                lt = str(self.plan_loss_type).lower()
                if lt == "infonce":
                    plan_loss = compute_infonce()
                elif lt == "cos":
                    plan_loss = compute_cos()
                elif lt == "mse":
                    plan_loss = compute_mse()
                elif lt == "cos_mse":
                    plan_loss = self.plan_cos_w * compute_cos() + self.plan_mse_w * compute_mse()
                elif lt == "infonce_cos":
                    plan_loss = compute_infonce() + self.plan_cos_w * compute_cos()
                elif lt == "infonce_mse":
                    plan_loss = compute_infonce() + self.plan_mse_w * compute_mse()
                else:
                    raise ValueError(f"Unknown PLAN_LOSS_TYPE={lt}")

                output["plan_loss"] = plan_loss
                output["total"] = output["total"] + self.plan_loss_w * plan_loss
        else:
            output["plan_loss"] = output["total"].sum() * 0.0


        # --- freq regression loss (S/D, 6 bands) ---
        if self.use_freq and (self.net._freq_pred is not None):
            # x_start is GT target (B,T,2*D)
            xa = x_start[..., :self.nfeats][..., :self.freq_feat_dim]  # (B,T,Dfeat)
            xb = x_start[..., self.nfeats: self.nfeats*2][..., :self.freq_feat_dim]

            S = 0.5 * (xa + xb)
            Drel = (xa - xb)

            if tgt_mask is not None:
                m = tgt_mask[..., 0].to(dtype=S.dtype)  # (B,T)
                S = S * m[:, :, None]
                Drel = Drel * m[:, :, None]

            Cs = dct_ii(S, dim=1, norm="ortho")
            Cd = dct_ii(Drel, dim=1, norm="ortho")

            (l0,l1),(m0,m1),(h0,h1) = default_bands(T, self.freq_r_low, self.freq_r_mid, self.freq_r_hi)

            Fs_low  = band_energy(Cs, l0, l1)
            Fs_mid  = band_energy(Cs, m0, m1)
            Fs_high = band_energy(Cs, h0, h1)
            Fd_low  = band_energy(Cd, l0, l1)
            Fd_mid  = band_energy(Cd, m0, m1)
            Fd_high = band_energy(Cd, h0, h1)

            gt = torch.stack([Fs_low, Fs_mid, Fs_high, Fd_low, Fd_mid, Fd_high], dim=1)  # (B,6,Dfeat)
            pred = self.net._freq_pred  # (B,6,Dfeat)

            # weights: high down-weighted
            w = torch.tensor([1.0, 1.0, self.freq_high_w, 1.0, 1.0, self.freq_high_w],
                             device=pred.device, dtype=pred.dtype).view(1, 6, 1)

            if keep_mask is not None:
                keep = (keep_mask.view(-1) > 0.5)  # (B,)
                if keep.any():
                    freq_loss = ((pred[keep] - gt[keep]) ** 2 * w).mean()
                else:
                    freq_loss = pred.sum() * 0.0
            else:
                freq_loss = ((pred - gt) ** 2 * w).mean()

            output["freq_loss"] = freq_loss
            output["total"] = output["total"] + self.freq_loss_w * freq_loss
        else:
            output["freq_loss"] = output["total"].sum() * 0.0


        return output

    def forward(self, batch):
        """
        Sampling: x_start=None as requested.
        """
        cond = batch["cond"]
        sources = batch["sources"]
        B = cond.shape[0]
        T = int(batch["motion_lens"][0])

        # source emb
        src_mask = self.generate_src_mask(T, batch["source_lens"]).to(sources.device)
        source_emb = self.motion_encoder(sources, src_mask[..., 0])

        timestep_respacing = self.sampling_strategy
        self.diffusion_test = MotionDiffusion(
            use_timesteps=space_timesteps(self.diffusion_steps, timestep_respacing),
            betas=self.betas,
            motion_rep=self.motion_rep,
            model_mean_type=ModelMeanType.START_X,
            model_var_type=ModelVarType.FIXED_SMALL,
            loss_type=LossType.MSE,
            rescale_timesteps=False,
        )

        # choose 3-branch CFG
        # s_text = float(getattr(self.cfg, "CFG_SCALE_TEXT", self.cfg_weight))
        # s_motion = float(getattr(self.cfg, "CFG_SCALE_MOTION", self.cfg_weight))
        # self.cfg_model = ClassifierFreeSampleModel(self.net, s_text, s_motion)
        # or 2-branch CFG
        self.cfg_model = ClassifierFreeSampleModel(self.net, self.cfg_weight)

        output = self.diffusion_test.ddim_sample_loop(
            self.cfg_model,
            (B, T, self.nfeats * 2),
            clip_denoised=False,
            progress=True,
            model_kwargs={
                "mask": None,
                "cond": cond,
                "source_emb": source_emb,
            },
            x_start=None
        )
        return {"output": output}
