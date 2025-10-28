from torch import nn
import torch
from torch.nn import TransformerEncoder, TransformerEncoderLayer

"""Two contrastive encoders"""
class TFC(nn.Module):
    def __init__(self, backbone, args, dim=256):
        super(TFC, self).__init__()

        self.backbone = backbone
        self.out_dim = dim

        self.projector_t = nn.Sequential(
            nn.Linear(args.TSlength_aligned* args.n_feature, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )

        self.projector_f = nn.Sequential(
            nn.Linear(args.TSlength_aligned * args.n_feature, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )


    def forward(self, x_in_t, x_in_f):
        # x_in_t: [B, T, C], x_in_f: [B, T_F, C]
        x, f = self.backbone(x_in_t, x_in_f)

        """Use Transformer"""
        # x = self.transformer_encoder_t(x_in_t)
        h_time = x.reshape(x.shape[0], -1)

        """Cross-space projector"""
        z_time = self.projector_t(h_time)

        """Frequency-based contrastive encoder"""
        # f = self.transformer_encoder_f(x_in_f)
        h_freq = f.reshape(f.shape[0], -1)

        """Cross-space projector"""
        z_freq = self.projector_f(h_freq)

        return h_time, z_time, h_freq, z_freq