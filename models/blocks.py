from .layers import *
# from mmcls_custom.models.backbones.vrwkv6 import RWKV_Block
# from mmcls_custom.models.backbones_T.vrwkv6_T import RWKV_Block_T
import torch
import torch.nn as nn

class TransformerBlock(nn.Module):
    def __init__(self,
                 latent_dim=512,
                 num_heads=8,
                 ff_size=1024,
                 dropout=0.,
                 cond_abl=False,
                 **kargs):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.cond_abl = cond_abl

        self.sa_block = VanillaSelfAttention(latent_dim, num_heads, dropout)
        self.ca_block = VanillaCrossAttention(latent_dim, latent_dim, num_heads, dropout, latent_dim)
        self.ffn = FFN(latent_dim, ff_size, dropout, latent_dim)

    def forward(self, x, y, emb=None, key_padding_mask=None):
        h1 = self.sa_block(x, emb, key_padding_mask)
        h1 = h1 + x
        h2 = self.ca_block(h1, y, emb, key_padding_mask)
        h2 = h2 + h1
        out = self.ffn(h2, emb)
        out = out + h2
        return out


class InterEditTransformerBlock(nn.Module):
    def __init__(self,
                 latent_dim=512,
                 num_heads=8,
                 ff_size=1024,
                 dropout=0.,
                 cond_abl=False,
                 cur_layer=0,
                 **kargs):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.cond_abl = cond_abl

        self.sa_block = VanillaSelfAttention(latent_dim * 2, num_heads, dropout, embed_dim=latent_dim)
        self.ffn = FFN(latent_dim * 2, ff_size, dropout, latent_dim)

        self.LPA = kargs.get('LPA', False)
        if self.LPA:
            from mmcls_custom.models.backbones_T.resnet import Resnet1D
            self.linear = nn.Linear(2 * latent_dim, latent_dim)
            self.conv = Resnet1D(latent_dim, kargs['cfg'].conv_layers, kargs['cfg'].dilation_rate,
                                 norm=kargs['cfg'].norm, first=(cur_layer == 0))

    def forward(self, x, y, plan=None, freq=None, emb=None, key_padding_mask=None):
        """
        x,y:   (B,T,C)
        plan:  (B,Kp,2C) or None
        freq:  (B,Kf,2C) or None
        emb:   (B,C)
        key_padding_mask: (B,T) True=pad
        """
        b, t, c = x.size(0), x.size(1), x.size(2)
        Kp = 0 if plan is None else plan.size(1)
        Kf = 0 if freq is None else freq.size(1)

        # (B,2T,2C) interleaved
        inputs = torch.empty((b, t * 2, 2 * c), device=x.device)
        inputs[:, ::2, :c] = x
        inputs[:, 1::2, :c] = y
        inputs[:, ::2, c:] = y
        inputs[:, 1::2, c:] = x

        mask = torch.ones((b, t * 2), dtype=torch.bool, device=x.device)
        index = torch.arange(t * 2, device=x.device).unsqueeze(0).repeat(b, 1)
        valid_idx = (~key_padding_mask).sum(-1) * 2
        mask[index < valid_idx.unsqueeze(1)] = False

        if (plan is not None) and (freq is not None):
            ctrl = torch.cat([plan, freq], dim=1)  # (B,Kp+Kf,2C)
        elif plan is not None:
            ctrl = plan                            # (B,Kp,2C)
        elif freq is not None:
            ctrl = freq                            # (B,Kf,2C)
        else:
            ctrl = None

        if ctrl is None:
            inputs_cat = inputs                    # (B,2T,2C)
            mask_cat = mask                        # (B,2T)
        else:
            inputs_cat = torch.cat([inputs, ctrl], dim=1)  # (B,2T+K,2C)
            mask_cat = torch.cat(
                [mask, torch.zeros((b, ctrl.size(1)), dtype=torch.bool, device=x.device)],
                dim=1
            )

        # SA
        h1 = self.sa_block(inputs_cat, emb, mask_cat)
        h1 = h1 + inputs_cat

        # split frames / ctrl
        h_frames = h1[:, :t * 2, :]               # (B,2T,2C)
        h_ctrl = None if ctrl is None else h1[:, t * 2:, :]  # (B,K,2C) or None

        # split plan/freq inside ctrl
        if h_ctrl is not None:
            if (Kp > 0) and (Kf > 0):
                h_plan = h_ctrl[:, :Kp, :]
                h_freq = h_ctrl[:, Kp:, :]
            elif Kp > 0:
                h_plan = h_ctrl
                h_freq = None
            else:
                h_plan = None
                h_freq = h_ctrl
        else:
            h_plan, h_freq = None, None

        if self.LPA:
            scan_1, scan_2 = h_frames[:, :, :c], h_frames[:, :, c:]
            outputs = torch.empty((b, t * 2, 2 * c), device=x.device)

            x_ = x.clone()
            y_ = y.clone()
            x_[key_padding_mask] = 0.0
            y_[key_padding_mask] = 0.0
            out_conv_x = self.conv(x_, emb)  # (B,T,C)
            out_conv_y = self.conv(y_, emb)

            final_out_x = self.linear(torch.cat((scan_1[:, ::2] + scan_2[:, 1::2], out_conv_x), -1))
            final_out_y = self.linear(torch.cat((scan_2[:, ::2] + scan_1[:, 1::2], out_conv_y), -1))

            outputs[:, ::2, :c] = final_out_x
            outputs[:, 1::2, :c] = final_out_y
            outputs[:, ::2, c:] = final_out_y
            outputs[:, 1::2, c:] = final_out_x

            out_frames = self.ffn(outputs, emb) + outputs

            # ctrl FFN: only if ctrl exists
            out_plan = None if h_plan is None else (self.ffn(h_plan, emb) + h_plan)
            out_freq = None if h_freq is None else (self.ffn(h_freq, emb) + h_freq)

            scan_1, scan_2 = out_frames[:, :, :c], out_frames[:, :, c:]
            x_new = scan_1[:, ::2] + scan_2[:, 1::2]
            y_new = scan_2[:, ::2] + scan_1[:, 1::2]
            return x_new, y_new, out_plan, out_freq

        else:
            # FFN on frames and ctrl together if ctrl exists, else only frames
            if h_ctrl is None:
                out_frames = self.ffn(h_frames, emb) + h_frames
                out_plan, out_freq = None, None
            else:
                h_cat = torch.cat([h_frames, h_ctrl], dim=1)
                out_cat = self.ffn(h_cat, emb) + h_cat
                out_frames = out_cat[:, :t * 2, :]
                out_ctrl = out_cat[:, t * 2:, :]

                if (Kp > 0) and (Kf > 0):
                    out_plan = out_ctrl[:, :Kp, :]
                    out_freq = out_ctrl[:, Kp:, :]
                elif Kp > 0:
                    out_plan = out_ctrl
                    out_freq = None
                else:
                    out_plan = None
                    out_freq = out_ctrl

            scan_1, scan_2 = out_frames[:, :, :c], out_frames[:, :, c:]
            x_new = scan_1[:, ::2] + scan_2[:, 1::2]
            y_new = scan_2[:, ::2] + scan_1[:, 1::2]
            return x_new, y_new, out_plan, out_freq
