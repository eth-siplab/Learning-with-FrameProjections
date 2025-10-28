import torch
import numpy as np
from torch.nn import Softmax
import torch.nn as nn
import torch.distributed as dist

class NTXentLoss(nn.Module):
    def __init__(self, batch_size, temperature, world_size):
        super(NTXentLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.world_size = world_size

        self.mask = self.mask_correlated_samples(batch_size, world_size)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def mask_correlated_samples(self, batch_size, world_size):
        N = 2 * batch_size * world_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size * world_size):
            mask[i, batch_size * world_size + i] = 0
            mask[batch_size * world_size + i, i] = 0
        return mask

    def forward(self, z_i, z_j):
        """
        Compute a distributed contrastive loss using a "local x global" similarity.
        z_i and z_j are embeddings from two views with shape (B, D) on each process.
        In a distributed setting, each process has a local batch of size B.
        After gathering, the global batch size is B_total = B * world_size.
        For a local process with rank r, its local samples correspond to global indices:
            r * B, r * B + 1, ..., r * B + (B-1)
        The loss for each local sample is computed by comparing its similarity (from one view)
        against the entire global set from the other view.
        """
        B = z_i.shape[0]
        device = z_i.device
        # If distributed, gather embeddings from all processes for one view.
        if dist.is_initialized():
            # Gather global embeddings for z_i and z_j.
            gathered_z_i = torch.cat(GatherLayer.apply(z_i), dim=0)  # shape: (global_B, D)
            gathered_z_j = torch.cat(GatherLayer.apply(z_j), dim=0)  # shape: (global_B, D)
            global_B = gathered_z_i.shape[0]
            local_rank = dist.get_rank()
        else:
            gathered_z_i = z_i
            gathered_z_j = z_j
            global_B = B
            local_rank = 0

        # Compute similarity between local z_i and global gathered_z_j.
        # This yields a matrix of shape (B, global_B)
        sim_matrix1 = self.similarity_f(z_i.unsqueeze(1), gathered_z_j.unsqueeze(0)) / self.temperature
        # Similarly, compute similarity between local z_j and global gathered_z_i.
        sim_matrix2 = self.similarity_f(z_j.unsqueeze(1), gathered_z_i.unsqueeze(0)) / self.temperature

        # Determine the positive indices for this process.
        # For a process with rank r, its local samples correspond to global indices r*B ... r*B+B-1.
        pos_idx = torch.arange(local_rank * B, local_rank * B + B, device=device)

        # For sim_matrix1: for each local sample i (i=0...B-1), the positive is at index (r*B + i) in the global gathered set.
        # Since sim_matrix1 has shape (B, global_B), we extract:
        pos_sim1 = sim_matrix1[torch.arange(B, device=device), pos_idx]  # shape: (B,)
        pos_sim2 = sim_matrix2[torch.arange(B, device=device), pos_idx]  # shape: (B,)
        # Concatenate the positives from both views.
        positives = torch.cat([pos_sim1, pos_sim2], dim=0).unsqueeze(1)  # shape: (2B, 1)

        # For negatives, we want all other similarities.
        # For sim_matrix1, create a mask that zeroes out the positive index for each row.
        mask1 = torch.ones_like(sim_matrix1, dtype=torch.bool)
        mask1[torch.arange(B, device=device), pos_idx] = False
        negatives1 = sim_matrix1[mask1].view(B, global_B - 1)

        mask2 = torch.ones_like(sim_matrix2, dtype=torch.bool)
        mask2[torch.arange(B, device=device), pos_idx] = False
        negatives2 = sim_matrix2[mask2].view(B, global_B - 1)

        negatives = torch.cat([negatives1, negatives2], dim=0)  # shape: (2B, global_B - 1)

        # Build logits by concatenating positive similarity and negatives.
        logits = torch.cat((positives, negatives), dim=1)
        # The target for each sample is that the correct similarity (the positive) is at index 0.
        labels = torch.zeros(2 * B, device=device).long()

        loss = self.criterion(logits, labels)
        # Normalize by the number of local samples.
        return loss / (2 * B)

class Cont_InfoNCE(torch.nn.Module):

    def __init__(self, device, batch_size, temperature=0.1):
        super(Cont_InfoNCE, self).__init__()
        self.temperature = temperature
        self.device = device
        self.softmax = torch.nn.Softmax(dim=-1)
        self.similarity_function = self._max_cross_corr
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    def _max_cross_corr(self, feats_1, feats_2):
        # feats_1: 1 x T (# time stamp)
        # feats_2: M (# aug) x T (# time stamp)
        feats_2 = feats_2.to(feats_1.dtype)
        feats_1 = feats_1 - torch.mean(feats_1, dim=-1, keepdim=True)
        feats_2 = feats_2 - torch.mean(feats_2, dim=-1, keepdim=True)

        min_N = min(feats_1.shape[-1], feats_2.shape[-1])
        padded_N = max(feats_1.shape[-1], feats_2.shape[-1]) * 2
        feats_1_pad = torch.nn.functional.pad(feats_1, (0, padded_N - feats_1.shape[-1]))
        feats_2_pad = torch.nn.functional.pad(feats_2, (0, padded_N - feats_2.shape[-1]))
        feats_1_fft = torch.fft.rfft(feats_1_pad)
        feats_2_fft = torch.fft.rfft(feats_2_pad)
        X = feats_1_fft * torch.conj(feats_2_fft)

        power_norm = (torch.std(feats_1, dim=-1, keepdim=True) *
                    torch.std(feats_2, dim=-1, keepdim=True)).to(X.dtype)
        power_norm = torch.where(power_norm == 0, torch.ones_like(power_norm), power_norm)
        X = X / power_norm

        cc = torch.fft.irfft(X) / (min_N - 1)
        max_cc = torch.max(cc, dim=-1).values

        return max_cc

    def forward(self, zis, zjs, speeds):
        """
        zis: M (# aug) x T (# time stamp)
        zjs: M (# aug) x T (# time stamp)
        """
        # Calculate distance for a single row of x.
        def per_x_dist(i):
            return self.similarity_function(zis[i:(i + 1), :], zjs)

        # Compute and stack distances for all rows of x.
        dist = torch.stack([per_x_dist(i) for i in range(zis.shape[0])])
        loss = self.criterion(dist, speeds)

        return loss 
    
def label_distance(labels_1, labels_2, dist_fn='l1', label_temperature=0.1):
    # labels: bsz x M(#augs)
    # output: bsz x M(#augs) x M(#augs)
    if dist_fn == 'l1':
        dist_mat = - torch.abs(labels_1[:, :, None] - labels_2[:, None, :])
    elif dist_fn == 'l2':
        dist_mat = - torch.abs(labels_1[:, :, None] - labels_2[:, None, :]) ** 2
    elif dist_fn == 'sqrt':
        dist_mat = - torch.abs(labels_1[:, :, None] - labels_2[:, None, :]).sqrt()
    else:
        raise NotImplementedError(f"`{dist_fn}` not implemented.")

    prob_mat = torch.nn.functional.softmax(dist_mat / label_temperature, dim=-1)
    return prob_mat

class CLIP_loss(nn.Module):
    def __init__(self, device, temperature=0.1):
        super(CLIP_loss, self).__init__()
        self.device = device
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss(reduction='mean')

    def forward(self, zis, zjs):
        """
        Args:
            zis: Tensor of shape (N, D), e.g. image (time) embeddings
            zjs: Tensor of shape (N, D), e.g. text (frequency) embeddings
        Returns:
            Scalar CLIP loss
        """
        # L2-normalize both sets of embeddings
        zis = torch.nn.functional.normalize(zis, p=2, dim=1)
        zjs = torch.nn.functional.normalize(zjs, p=2, dim=1)

        # Compute cosine similarity and scale
        logits = (zis @ zjs.T) / self.temperature

        # Ground-truth labels are 0, 1, ..., N-1
        labels = torch.arange(zis.size(0), device=self.device)

        # Cross-entropy loss in both directions
        loss_i2t = self.criterion(logits, labels)      # image → text
        loss_t2i = self.criterion(logits.T, labels)    # text → image

        return (loss_i2t + loss_t2i) / 2

class MTM_loss(nn.Module):
    def __init__(self, device, args=None):
        super(MTM_loss, self).__init__()
        self.device = device
        self.args = args
        self.criterion = nn.CrossEntropyLoss(reduction='mean')
        self.awl = AutomaticWeightedLoss(2)
        self.contrastive = ContrastiveWeight(self.args)
        self.aggregation = AggregationRebuild(self.args)
        self.head = nn.Linear(128, self.args.len_sw * self.args.n_feature).to(self.device)  
        self.mse = torch.nn.MSELoss()        

    def forward(self, z, h, x):


        loss_cl, similarity_matrix, logits, positives_mask = self.contrastive(z)
        rebuild_weight_matrix, agg_x = self.aggregation(similarity_matrix, h)

        pred_x = self.head(agg_x.reshape(agg_x.size(0), -1))
        loss_rb = self.mse(pred_x, x.reshape(x.size(0), -1).detach())
        loss = self.awl(loss_cl, loss_rb)

        return loss

class AutomaticWeightedLoss(torch.nn.Module):
    """automatically weighted multi-task loss
    Params：
        num: int，the number of loss
        x: multi-task loss
    Examples：
        loss1=1
        loss2=2
        awl = AutomaticWeightedLoss(2)
        loss_sum = awl(loss1, loss2)
    """

    def __init__(self, num=2):
        super(AutomaticWeightedLoss, self).__init__()
        params = torch.ones(num, requires_grad=True)
        self.params = torch.nn.Parameter(params)

    def forward(self, *x):
        loss_sum = 0
        for i, loss in enumerate(x):
            loss_sum += 0.5 / (self.params[i] ** 2) * loss + torch.log(1 + self.params[i] ** 2)
        return loss_sum

class ContrastiveWeight(torch.nn.Module):

    def __init__(self, args):
        super(ContrastiveWeight, self).__init__()
        self.temperature = 0.2

        self.bce = torch.nn.BCELoss()
        self.softmax = torch.nn.Softmax(dim=-1)
        self.log_softmax = torch.nn.LogSoftmax(dim=-1)
        self.kl = torch.nn.KLDivLoss(reduction='batchmean')
        self.positive_nums = args.positive_nums

    def get_positive_and_negative_mask(self, similarity_matrix, cur_batch_size):
        diag = np.eye(cur_batch_size)
        mask = torch.from_numpy(diag)
        mask = mask.type(torch.bool)

        oral_batch_size = cur_batch_size // (self.positive_nums + 1)

        positives_mask = np.zeros(similarity_matrix.size())
        for i in range(self.positive_nums + 1):
            ll = np.eye(cur_batch_size, cur_batch_size, k=oral_batch_size * i)
            lr = np.eye(cur_batch_size, cur_batch_size, k=-oral_batch_size * i)
            positives_mask += ll
            positives_mask += lr

        positives_mask = torch.from_numpy(positives_mask).to(similarity_matrix.device)
        positives_mask[mask] = 0

        negatives_mask = 1 - positives_mask
        negatives_mask[mask] = 0

        return positives_mask.type(torch.bool), negatives_mask.type(torch.bool)

    def forward(self, batch_emb_om):
        cur_batch_shape = batch_emb_om.shape

        # get similarity matrix among mask samples
        norm_emb = torch.nn.functional.normalize(batch_emb_om, dim=1)
        similarity_matrix = torch.matmul(norm_emb, norm_emb.transpose(0, 1))

        # get positives and negatives similarity
        positives_mask, negatives_mask = self.get_positive_and_negative_mask(similarity_matrix, cur_batch_shape[0])

        positives = similarity_matrix[positives_mask].view(cur_batch_shape[0], -1)
        negatives = similarity_matrix[negatives_mask].view(cur_batch_shape[0], -1)

        # generate predict and target probability distributions matrix
        logits = torch.cat((positives, negatives), dim=-1)
        y_true = torch.cat(
            (torch.ones(cur_batch_shape[0], positives.shape[-1]), torch.zeros(cur_batch_shape[0], negatives.shape[-1])),
            dim=-1).to(batch_emb_om.device).float()

        # multiple positives - KL divergence
        predict = self.log_softmax(logits / self.temperature)
        loss = self.kl(predict, y_true)
        
        return loss, similarity_matrix, logits, positives_mask

class AggregationRebuild(torch.nn.Module):

    def __init__(self, args):
        super(AggregationRebuild, self).__init__()
        self.args = args
        self.temperature = 0.2
        self.softmax = torch.nn.Softmax(dim=-1)
        self.mse = torch.nn.MSELoss()

    def forward(self, similarity_matrix, batch_emb_om):
        cur_batch_shape = batch_emb_om.shape

        # get the weight among (oral, oral's masks, others, others' masks)
        similarity_matrix /= self.temperature

        similarity_matrix = similarity_matrix - torch.eye(cur_batch_shape[0]).to(
            similarity_matrix.device).float() * 1e12
        rebuild_weight_matrix = self.softmax(similarity_matrix)

        batch_emb_om = batch_emb_om.reshape(cur_batch_shape[0], -1)

        # generate the rebuilt batch embedding (oral, others, oral's masks, others' masks)
        rebuild_batch_emb = torch.matmul(rebuild_weight_matrix, batch_emb_om)

        # get oral' rebuilt batch embedding
        rebuild_oral_batch_emb = rebuild_batch_emb.reshape(cur_batch_shape[0], cur_batch_shape[1], -1)

        return rebuild_weight_matrix, rebuild_oral_batch_emb        