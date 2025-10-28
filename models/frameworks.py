import copy
import random
from functools import wraps

import torch
from torch import nn
import torch.nn.functional as F
from .backbones import *
from .models_nc import *
from .TC import *
from utils import WaveletTransform, FourierTransform
from data_preprocess import augmentations

class SimCLR(nn.Module):
    def __init__(self, backbone, dim=128):
        super(SimCLR, self).__init__()

        self.encoder = backbone
        self.bb_dim = self.encoder.out_dim
        self.projector = Projector(model='SimCLR', bb_dim=self.bb_dim, prev_dim=self.bb_dim, dim=dim)

    def forward(self, x1, x2,  DACL_training=False):
        if self.encoder.__class__.__name__ in ['AE', 'CNN_AE']:
            x1_encoded, z1 = self.encoder(x1)
            x2_encoded, z2 = self.encoder(x2)
        else:
            _, z1 = self.encoder(x1)
            _, z2 = self.encoder(x2)

        if len(z1.shape) == 3:
            z1 = z1.reshape(z1.shape[0], -1)
            z2 = z2.reshape(z2.shape[0], -1)

        z1 = self.projector(z1)
        z2 = self.projector(z2)

        if self.encoder.__class__.__name__ in ['AE', 'CNN_AE']:
            return x1_encoded, x2_encoded, z1, z2
        else:
            return z1, z2

class NNCLR(nn.Module):
    def __init__(self, backbone, dim=128, pred_dim=64):
        super(NNCLR, self).__init__()
        self.encoder = backbone
        self.bb_dim = self.encoder.out_dim
        self.projector = Projector(model='NNCLR', bb_dim=self.bb_dim, prev_dim=self.bb_dim, dim=dim)
        self.predictor = Predictor(model='NNCLR', dim=dim, pred_dim=pred_dim)

    def forward(self, x1, x2):
        if self.encoder.__class__.__name__ in ['AE', 'CNN_AE']:
            x1_encoded, z1 = self.encoder(x1)
            x2_encoded, z2 = self.encoder(x2)
        else:
            _, z1 = self.encoder(x1)
            _, z2 = self.encoder(x2)

        if len(z1.shape) == 3:
            z1 = z1.reshape(z1.shape[0], -1)
            z2 = z2.reshape(z2.shape[0], -1)
        
        z1 = self.projector(z1)
        z2 = self.projector(z2)

        p1 = self.predictor(z1)
        p2 = self.predictor(z2)

        if self.encoder.__class__.__name__ in ['AE', 'CNN_AE']:
            return x1_encoded, x2_encoded, p1, p2, z1.detach(), z2.detach()
        else:
            return p1, p2, z1.detach(), z2.detach()  

class BYOL(nn.Module):
    def __init__(
        self,
        DEVICE,
        backbone,
        window_size = 30,
        n_channels = 77,
        hidden_layer = -1,
        projection_size = 64,
        projection_hidden_size = 256,
        moving_average = 0.99,
        use_momentum = True,
    ):
        super().__init__()

        net = backbone
        self.bb_dim = net.out_dim
        self.online_encoder = NetWrapper(net, projection_size, projection_hidden_size, DEVICE=DEVICE, layer=hidden_layer)

        self.use_momentum = use_momentum
        self.target_encoder = None
        self.target_ema_updater = EMA(moving_average)

        self.online_predictor = Predictor(model='byol', dim=projection_size, pred_dim=projection_hidden_size)

        self.to(DEVICE)

        # send a mock image tensor to instantiate singleton parameters
        self.forward(torch.randn(2, window_size, n_channels, device=DEVICE),
                     torch.randn(2, window_size, n_channels, device=DEVICE))

    @singleton('target_encoder')
    def _get_target_encoder(self):
        target_encoder = copy.deepcopy(self.online_encoder)
        for p in target_encoder.parameters():
            p.requires_grad = False
        return target_encoder

    def reset_moving_average(self):
        del self.target_encoder
        self.target_encoder = None

    def update_moving_average(self):
        assert self.target_encoder is not None, 'target encoder has not been created yet'
        update_moving_average(self.target_ema_updater, self.target_encoder, self.online_encoder)

    def forward(
        self,
        x1,
        x2,
        return_embedding = False,
        return_projection = True,
        require_lat = False
    ):
        assert not (self.training and x1.shape[0] == 1), 'you must have greater than 1 sample when training, due to the batchnorm in the projection layer'

        if return_embedding:
            return self.online_encoder(x1, return_projection = return_projection)

        if self.online_encoder.net.__class__.__name__ in ['AE', 'CNN_AE']:
            online_proj_one, x1_decoded, lat1 = self.online_encoder(x1)
            online_proj_two, x2_decoded, lat2 = self.online_encoder(x2)
        else:
            online_proj_one, lat1 = self.online_encoder(x1)
            online_proj_two, lat2 = self.online_encoder(x2)

        online_pred_one = self.online_predictor(online_proj_one)
        online_pred_two = self.online_predictor(online_proj_two)

        with torch.no_grad():
            target_encoder = self._get_target_encoder() if self.use_momentum else self.online_encoder
            if self.online_encoder.net.__class__.__name__ in ['AE', 'CNN_AE']:
                target_proj_one, _, _ = target_encoder(x1)
                target_proj_two, _, _ = target_encoder(x2)
            else:
                target_proj_one, _ = target_encoder(x1)
                target_proj_two, _ = target_encoder(x2)
            
            target_proj_one.detach_()
            target_proj_two.detach_()

        if self.online_encoder.net.__class__.__name__ in ['AE', 'CNN_AE']:
            if require_lat:
                return x1_decoded, x2_decoded, online_pred_one, online_pred_two, target_proj_one.detach(), target_proj_two.detach(), lat1, lat2
            else:
                return x1_decoded, x2_decoded, online_pred_one, online_pred_two, target_proj_one.detach(), target_proj_two.detach()
        else:
            if require_lat:
                return online_pred_one, online_pred_two, target_proj_one.detach(), target_proj_two.detach(), lat1, lat2
            else:
                return online_pred_one, online_pred_two, target_proj_one.detach(), target_proj_two.detach()

class TSTCC(nn.Module):
    def __init__(self, backbone, DEVICE, temp_unit='tsfm', tc_hidden=100):
        """
        dim: feature dimension (default: 2048)
        pred_dim: hidden dimension of the predictor (default: 512)
        """
        super(TSTCC, self).__init__()
        self.encoder = backbone
        self.bb_dim = self.encoder.out_channels
        self.TC = TC(self.bb_dim, DEVICE, tc_hidden=tc_hidden, temp_unit=temp_unit).to(DEVICE)
        self.projector = Projector(model='TS-TCC', bb_dim=self.bb_dim, prev_dim=None, dim=tc_hidden)

    def forward(self, x1, x2, DACL_training=False):
        """
        Input:
            x1: first views of images
            x2: second views of images
        Output:
            p1, p2, z1, z2: predictors and targets of the network
            See Sec. 3 of https://arxiv.org/abs/2011.10566 for detailed notations
        """
            
        _, z1 = self.encoder(x1)
        _, z2 = self.encoder(x2)

        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

        nce1, c_t1 = self.TC(z1, z2)
        nce2, c_t2 = self.TC(z2, z1)

        p1 = self.projector(c_t1)
        p2 = self.projector(c_t2)

        return nce1, nce2, p1, p2
    
"""
https://github.com/facebookresearch/vicreg/blob/main/main_vicreg.py

"""
class VICReg(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.num_features = int(args.p)

    def forward(self, x, y):
        repr_loss = torch.nn.functional.mse_loss(x, y)

        x = x - x.mean(dim=0)
        y = y - y.mean(dim=0)

        std_x = torch.sqrt(x.var(dim=0) + 0.0001)
        std_y = torch.sqrt(y.var(dim=0) + 0.0001)
        std_loss = torch.mean(torch.nn.functional.relu(1 - std_x)) / 2 + torch.mean(torch.nn.functional.relu(1 - std_y)) / 2

        cov_x = (x.T @ x) / (self.args.batch_size - 1)
        cov_y = (y.T @ y) / (self.args.batch_size - 1)
        cov_loss = off_diagonal(cov_x).pow_(2).sum().div(
            self.num_features
        ) + off_diagonal(cov_y).pow_(2).sum().div(self.num_features)

        loss = (
            self.args.sim_coeff * repr_loss
            + self.args.std_coeff * std_loss
            + self.args.cov_coeff * cov_loss
        )
        return loss
    
def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

"""
https://github.com/facebookresearch/barlowtwins/blob/main/main.py

"""
class BarlowTwins(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        # normalization layer for the representations z1 and z2
        self.bn = nn.BatchNorm1d(args.p, affine=False).to(args.cuda)        

    def forward(self, z1, z2):
        c = self.bn(z1).T @ self.bn(z2)
        # sum the cross-correlation matrix between all gpus
        c.div_(self.args.batch_size)

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        loss = on_diag + self.args.lambd * off_diag
        return loss

class CLIP(nn.Module):
    def __init__(self, backbone, backbone_FT, DEVICE, dim=128, args=None):
        super(CLIP, self).__init__()
        self.encoder = backbone
        self.args = args
        self.encoder_FT = backbone_FT
        self.bb_dim = self.encoder.out_dim

        self.projector = Projector(model='CLIP', bb_dim=self.bb_dim, prev_dim=self.bb_dim, dim=dim)
        self.projector_FT = Projector(model='CLIP_FT', bb_dim=self.bb_dim, prev_dim=self.bb_dim, dim=dim)
        self.device = DEVICE
    
    def forward(self, x1, x2):
        x2 = x2.transpose(1, 2) 
        _, z1 = self.encoder(x1)
        z2 = self.encoder_FT(x2)

        if len(z1.shape) == 3:
            z1 = z1.reshape(z1.shape[0], -1)
            z2 = z2.reshape(z2.shape[0], -1)

        z1 = self.projector(z1)
        z2 = self.projector_FT(z2)

        return z1, z2

class MTM(nn.Module):
    def __init__(self, backbone, DEVICE, dim=128, args=None):
        super(MTM, self).__init__()
        self.encoder = backbone
        self.args = args
        self.bb_dim = self.encoder.out_dim 
        self.device = DEVICE
        self.projector = Projector(model='MTM', bb_dim=self.bb_dim, prev_dim=self.bb_dim, dim=dim)

    def forward(self, x1, x2):
        data_masked_om = torch.cat([x1, x2], 0) # data_masked_om = torch.cat([data, data_masked_m], 0)
        _, h = self.encoder(data_masked_om)
        z = self.projector(h)
        return z, h, data_masked_om        

class IsoAlign(nn.Module):
    def __init__(self, backbone, spect_encoder, FT_encoder, DEVICE, dim=128, batch_size=1024, args=None):
        super(IsoAlign, self).__init__()
        self.encoder = backbone
        self.args = args
        self.batch_size = batch_size
        self.spect_encoder = spect_encoder
        self.FT_encoder = FT_encoder
        self.bb_dim = self.encoder.out_dim 
        self.device = DEVICE
        self.projector = Projector(model='IsoAlign', bb_dim=self.bb_dim, prev_dim=self.bb_dim, dim=dim)
        self.projector_spect = Projector(model='IsoAlign', bb_dim=self.bb_dim, prev_dim=self.bb_dim, dim=dim)
        self.projector_FT = Projector(model='IsoAlign', bb_dim=self.bb_dim, prev_dim=self.bb_dim, dim=dim)

        # self.predictor_FT = nn.Linear(dim, dim) # Ablation
        self.predictor_FT = ConvMapping(in_channels=dim, hidden_channels=64, kernel_size=3).to(DEVICE)
        # self.predictor_FT = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(inplace=True), nn.Linear(dim, dim)) # Ablation

        # self.predictor_cwt = nn.Linear(dim, dim) # Ablation
        self.predictor_cwt = ConvMapping(in_channels=dim, hidden_channels=64, kernel_size=3).to(DEVICE)
        # self.predictor_cwt = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(inplace=True), nn.Linear(dim, dim)) # Ablation

        self.wavelet_transform = WaveletTransform(wavelet='cmor1-1', fs=25)
        self.FT_transform = FourierTransform(fs=25)

        self.mask_samples_from_same_repr = self._get_correlated_mask().type(torch.bool)
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")
        self.steps = 32

    def _get_correlated_mask(self):
        diag = np.eye(2 * self.batch_size)
        l1 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=-self.batch_size)
        l2 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=self.batch_size)
        mask = torch.from_numpy((diag + l1 + l2))
        mask = (1 - mask).type(torch.bool)
        return mask.to(self.device)      

    def cont_loss(self, r1, r2, batch_size, sim_matrix=None):
        representations = torch.cat([r1, r2], dim=0)

        similarity_matrix = torch.matmul(representations, representations.t())

        # Adjust the top-left quadrant using the original similarity for time series
        if sim_matrix is not None:
            similarity_matrix[:batch_size, :batch_size] *= sim_matrix       

        # filter out the scores from the positive samples
        l_pos = torch.diag(similarity_matrix, batch_size)
        r_pos = torch.diag(similarity_matrix, -batch_size)

        positives = torch.cat([l_pos, r_pos]).view(2 * batch_size, 1)

        negatives = similarity_matrix[self.mask_samples_from_same_repr].view(2 * batch_size, -1)

        logits = torch.cat((positives, negatives), dim=1)
        logits /= 0.15

        labels = torch.zeros(2 * batch_size).to(logits.device).long()
        loss = self.criterion(logits, labels)
        return loss / (2 * batch_size)

    def calc_loss(self, e1, e2, e3): # e1 -> Time, e2 -> CWT, e3 -> FT
        e1 = torch.nn.functional.normalize(e1, dim=1)
        e2 = torch.nn.functional.normalize(e2, dim=1)
        e3 = torch.nn.functional.normalize(e3, dim=1)

        predicted_FT_latent = self.predictor_FT(e1)
        predicted_cwt_latent = self.predictor_cwt(e1)

        loss_predicted_cwt = torch.nn.functional.l1_loss(predicted_cwt_latent, e2) / self.batch_size
        loss_predicted_FT = torch.nn.functional.l1_loss(predicted_FT_latent, e3) / self.batch_size

        # symmetric loss functions
        loss1 = self.cont_loss(e1, e2, self.batch_size) 
        loss2 = self.cont_loss(e1, e3, self.batch_size)
        loss3 = self.cont_loss(e2, e3, self.batch_size)

        # if self.args.wo_OB: # without ortohogonal base
        #     return loss1 + (loss_predicted_cwt)
        # elif self.args.wo_OF: # without overcomplete frames
        #     return loss2 + (loss_predicted_FT)
        # else: # usual
        #     return loss1 + loss2 + loss3 + (loss_predicted_FT + loss_predicted_cwt)

        return loss1 + loss2 + loss3 + (loss_predicted_FT + loss_predicted_cwt)

        # return loss1 + loss2 + loss3

    def forward(self, x1, x2, x3):
        B, C, T = x1.shape  # Original shape (B, C, T)

        x1 = x1.transpose(1, 2) # (B, C, T) -> (B, T, C)
        x2 = x2.permute(0, 3, 1, 2)  # (B, F, T, C) -> (B, C, F, T)
        # x3.shape = (B, C, F)

        _, R_t = self.encoder(x1)
        out, R_f = self.spect_encoder(x2)
        _, R_f_FT = self.FT_encoder(x3)

        R_t = self.projector(R_t)
        R_f = self.projector_spect(R_f)
        R_f_FT = self.projector_FT(R_f_FT)

        loss = self.calc_loss(R_t, R_f, R_f_FT)

        # Ablations

        # R_t = torch.nn.functional.normalize(R_t, dim=1)
        # R_f = torch.nn.functional.normalize(R_f, dim=1)
        # R_f_FT = torch.nn.functional.normalize(R_f_FT, dim=1)

        # predicted_FT_latent = self.predictor1(R_t)
        # predicted_cwt_latent = self.predictor2(R_t)

        # loss_predicted_FT = torch.nn.functional.l1_loss(predicted_FT_latent, R_f) 
        # loss_predicted_cwt = torch.nn.functional.l1_loss(predicted_cwt_latent, R_f_FT)

        # loss_consistency = torch.dist(self.predictor2.weight @ torch.linalg.inv(self.predictor1.weight), self.const.weight) / B

        # symmetric loss functions
        # loss1 = self.cont_loss(R_t, R_f, self.batch_size)
        # loss2 = self.cont_loss(R_t, R_f_FT, self.batch_size)
        # loss3 = self.cont_loss(R_f, R_f_FT, self.batch_size)

        # loss1_1 = torch.nn.functional.pairwise_distance(R_t, R_f, p=2).sum() / B
        # loss2_1 = torch.nn.functional.pairwise_distance(R_t, R_f_FT, p=2).sum() / B
        # loss3_1 = torch.nn.functional.pairwise_distance(R_f, R_f_FT, p=2).sum() / B

        # loss1 = self.cont_loss_negs(R_t, R_f_FT, self.batch_size, sim_matrix=sim_matrix_FT)
        # loss2 = self.cont_loss_negs(R_t, R_f, self.batch_size, sim_matrix=sim_matrix_FT)
        # loss3 = self.cont_loss_negs(R_f, R_f_FT, self.batch_size, sim_matrix=sim_matrix_FT)

        return loss
        # return loss1 + loss2 + loss3 + (loss_predicted_FT + loss_predicted_cwt)
        # return loss_predicted_FT + loss_predicted_cwt
        # return loss_global + loss1_1 + loss2_1 + loss3_1


class IntegratedEncoder(nn.Module):
    def __init__(self, model, map_ft, map_cwt, args, out_dim):
        super(IntegratedEncoder, self).__init__()
        self.encoder = model.encoder
        self.map_ft = map_ft  
        self.map_cwt = map_cwt 
        self.wo_OF = args.wo_OF
        self.wo_OB = args.wo_OB        
        self.out_dim = out_dim * 3 if not self.wo_OF and not self.wo_OB else out_dim * 2
        # self.out_dim = out_dim * 1 if not self.wo_OF and not self.wo_OB else out_dim * 2 # second ablation

    def forward(self, x_sample):
        _, time_emb = self.encoder(x_sample)

        # Map the embeddings using the learned mappers.
        ft_mapped  = self.map_ft(time_emb)
        cwt_mapped = self.map_cwt(time_emb)

        # Ablations
        # if self.wo_OF: # without overcomplete frames
        #     all_emb = torch.cat([ft_mapped, time_emb], dim=1)
        # elif self.wo_OB:
        #     all_emb = torch.cat([cwt_mapped, time_emb], dim=1)
        # else: # normal    
        #     all_emb = torch.cat([ft_mapped, time_emb, cwt_mapped], dim=1)

        all_emb = torch.cat([ft_mapped, time_emb, cwt_mapped], dim=1)

        # all_emb = time_emb

        return _, all_emb