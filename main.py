import os 
import argparse
import numpy as np
import torch
import random
from trainer_SSL_LE import *
from models.ts2vec import *

parser = argparse.ArgumentParser(description='argument setting of network')
parser.add_argument('--cuda', default=0, type=int, help='cuda device ID, 0/1')
parser.add_argument('--num_gpus', default=1, type=int, help='number of gpus')
parser.add_argument('--seed', default=10, type=int, help='seed')
# Hyperparameter
parser.add_argument('--batch_size', type=int, default=64, help='batch size of training')
parser.add_argument('--n_epoch', type=int, default=90, help='number of training epochs')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--lr_cls', type=float, default=1e-3, help='learning rate for linear classifier')
parser.add_argument('--scheduler', type=bool, default=True, help='if or not to use a scheduler')
parser.add_argument('--use_amp', type=bool, default=True, help='if or not to use automatic mixed precision')
parser.add_argument('--weight_decay', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)', dest='weight_decay')
#
# Datasets
parser.add_argument('--dataset', type=str, default='ucihar', choices=['ucihar', 'hhar', 'usc', 'clemson', 'ieee_small','ieee_big', 'dalia', 'chapman', 'cpsc', 'sleep'], help='name of dataset')
parser.add_argument('--n_feature', type=int, default=77, help='name of feature dimension')
parser.add_argument('--len_sw', type=int, default=30, help='length of sliding window')
parser.add_argument('--n_class', type=int, default=18, help='number of class')
parser.add_argument('--cases', type=str, default='random', choices=['random', 'subject', 'subject_large', 'cross_device', 'joint_device'],
                    help='name of scenarios, cross_device and joint_device only applicable when hhar is used')
parser.add_argument('--split_ratio', type=float, default=0.2, help='split ratio of test/val: train(0.64), val(0.16), test(0.2)')
parser.add_argument('--target_domain', type=str, default='0', help='the target domain, [0 to 29] for ucihar, '
                                                                   '[1,2,3,5,6,9,11,13,14,15,16,17,19,20,21,22,23,24,25,29] for shar, '
                                                                   '[a-i] for hhar')

# Augmentations
parser.add_argument('--aug1', type=str, default='jit_scal',
                    choices=['na', 'noise', 'scale', 'shift', 'negate', 'perm', 'shuffle', 't_flip', 't_warp', 'resample', 'random_out','rotation', 'perm_jit', 'jit_scal', 'hfc', 'lfc', 'p_shift', 'ap_p', 'ap_f', 'random'],
                    help='the type of augmentation transformation')
parser.add_argument('--aug2', type=str, default='resample',
                    choices=['na', 'noise', 'scale', 'shift', 'negate', 'perm', 'shuffle', 't_flip', 't_warp', 'resample', 'random_out', 'rotation', 'perm_jit', 'jit_scal', 'hfc', 'lfc', 'p_shift', 'ap_p', 'ap_f', 'random'],
                    help='the type of augmentation transformation')

parser.add_argument('--aug3', action='store_true', help='ablation') # ablation for multi view 

# Frameworks
parser.add_argument('--backbone', type=str, default='DCL', choices=['FCN', 'DCL', 'LSTM', 'AE', 'CNN_AE', 'Transformer', 'resnet', 'unet', 'TFC_backbone'], help='name of backbone network')
parser.add_argument('--out_dim', type=int, default=128, help='output dimension of the encoder')
parser.add_argument('--block', type=int, default=8, help='output dimension of the encoder')

parser.add_argument('--framework', type=str, default='byol', choices=['byol', 'simsiam', 'simclr', 'nnclr', 'tstcc', 'isoalign', 'vicreg', 'barlowtwins', 'ts2vec', 'tfc', 'clip', 'mtm'], help='name of framework')
parser.add_argument('--criterion', type=str, default='cos_sim', choices=['cos_sim', 'NTXent'], help='type of loss function for contrastive learning')

parser.add_argument('--p', type=int, default=128, help='byol: projector size, simsiam: projector output size, simclr: projector output size')
parser.add_argument('--phid', type=int, default=128, help='byol: projector hidden size, simsiam: predictor hidden size, simclr: na')

# byol
parser.add_argument('--lr_mul', type=float, default=10.0,
                    help='lr multiplier for the second optimizer when training byol')
parser.add_argument('--EMA', type=float, default=0.996, help='exponential moving average parameter')

# nnclr
parser.add_argument('--mmb_size', type=int, default=1024, help='maximum size of NNCLR support set')

# Isoalign ablations
parser.add_argument('--wo_OB', action='store_true', help='without Fourier transformation')
parser.add_argument('--wo_OF', action='store_true', help='without wavelet transformation')

# TS-TCC
parser.add_argument('--lambda1', type=float, default=1.0, help='weight for temporal contrastive loss')
parser.add_argument('--lambda2', type=float, default=1.0, help='weight for contextual contrastive loss, also used as the weight for reconstruction loss when AE or CAE being backbone network')
parser.add_argument('--temp_unit', type=str, default='tsfm', choices=['tsfm', 'lstm', 'blstm', 'gru', 'bgru'], help='temporal unit in the TS-TCC')

# simMTM
parser.add_argument('--masking_ratio', type=float, default=0.2, help='masking ratio')
parser.add_argument('--positive_nums', type=int, default=3, help='positive series numbers')
parser.add_argument('--lm', type=int, default=3, help='average length of masking subsequences')

# VicReg
parser.add_argument('--sim_coeff', type=float, default=25, help='weight for similarity loss')
parser.add_argument('--std_coeff', type=float, default=25, help='weight for standard deviation loss')
parser.add_argument('--cov_coeff', type=float, default=1, help='weight for covariance loss')

# Barlow Twins
parser.add_argument('--lambd', type=float, default=0.0051, help='weight for the off-diagonal terms in the covariance matrix')

# hhar
parser.add_argument('--device', type=str, default='Phones', choices=['Phones', 'Watch'], help='data of which device to use (random case); data of which device to be used as training data (cross-device case, data from the other device as test data)')

# plot
parser.add_argument('--plt', type=bool, default=False, help='if or not to plot results')
parser.add_argument('--plot_tsne', action='store_true', help='if or not to plot t-SNE')

# Example: python main.py --framework 'simclr' --backbone 'resnet' --dataset 'ieee_small' --n_epoch 256 --batch_size 1024 --lr 1e-3 --lr_cls 0.03 --cuda 0 --cases 'subject_large' --aug1 'perm_jit' --aug2 'perm_jit'

# Example: python main.py --framework 'isoalign' --backbone 'resnet' --dataset 'ieee_small' --n_epoch 256 --batch_size 1024 --lr 1e-3 --lr_cls 0.03 --cuda 0 --cases 'subject_large'

# python main.py --framework 'isoalign' --backbone 'resnet' --dataset 'clemson' --n_epoch 256 --batch_size 512 --lr 1e-3 --lr_cls 0.03 --cuda 0 --cases 'subject_large'

############### Parser done ################

# Domains for each dataset
def set_domain(args):
    dataset = args.dataset
    if dataset == 'ucihar':
        domain = [0, 1, 2, 3, 4]
    elif dataset == 'usc':
        domain = [10, 11, 12, 13]
    elif dataset == 'ieee_small':
        domain = [0, 1, 2, 3, 4]
    elif dataset == 'ieee_big':
        domain = [17, 18, 19, 20, 21]
    elif dataset == 'dalia':
        domain = [0, 1, 2, 3, 4] 
    elif dataset == 'clemson':
        domain = [i for i in range(0, 10)]
    elif dataset == 'chapman':
        domain = [3]
    elif dataset == 'cpsc':
        domain = [1]
    elif dataset == 'hhar':
        domain = ['a', 'b', 'c', 'd']
    elif dataset == 'sleep':
        domain = [0]
    return domain

# Set seed 
def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed) 
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

############### Rep done ################

def main_SSL_LE(args, DEVICE):
    setattr(args, 'cases', 'subject_large_ssl_fn') # Pretrain the models in the large unlabelled data 
    train_loaders, val_loader, test_loader = setup_dataloaders(args)

    if not args.framework == 'ts2vec':
        model, optimizers, schedulers, criterion, classifier, criterion_cls, optimizer_cls = setup(args, DEVICE)
        best_pretrain_model = train(train_loaders, val_loader, model, DEVICE, optimizers, schedulers, criterion, args)
        # Only rank 0 performs testing; other ranks skip testing and return None.
        if not ('LOCAL_RANK' in os.environ) or dist.get_rank() == 0:
            best_pretrain_model = test(train_loaders, best_pretrain_model, DEVICE, criterion, args)
        else:
            best_pretrain_model = None        
    else:
        _, _, _, _, classifier, criterion_cls, optimizer_cls = setup(args, DEVICE)
        best_pretrain_model = train_ts2vec(train_loaders, val_loader, DEVICE, args)

    ####################################################################################################################
    # For subsequent steps (e.g., linear evaluation), only rank 0 proceeds. No need multi-GPU for the rest.
    if 'LOCAL_RANK' in os.environ and dist.get_rank() != 0:
        # Non-rank-0 processes can skip further steps.
        return None

    if args.framework == 'isoalign':
        # _ = test_isoalign(train_loaders, best_pretrain_model, args)
        best_pretrain_model = train_mappers(train_loaders, best_pretrain_model, DEVICE, args)
  
    trained_backbone = lock_backbone(best_pretrain_model, args)  # Linear evaluation

    setattr(args, 'cases', 'subject') # Fine tune the models in the limited labelled data with the same target subject/domain
    train_loaders, val_loader, test_loader = setup_dataloaders(args)
    best_lincls = train_lincls(train_loaders, val_loader, trained_backbone, classifier, DEVICE, optimizer_cls, criterion_cls, args)
    error = test_lincls(test_loader, trained_backbone, best_lincls, DEVICE, criterion_cls, args)  # Evaluate with the target domain
    delete_files(args)
    return error

def main():
    args = parser.parse_args()
    set_seed(args.seed)
    
    # Determine whether to run distributed training by checking for LOCAL_RANK.
    if 'LOCAL_RANK' in os.environ:
        import torch.distributed as dist
        dist.init_process_group(backend='nccl')
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        DEVICE = torch.device("cuda", local_rank)
        args.num_gpus = torch.cuda.device_count()
        setattr(args, 'rank', local_rank)
        distributed = True
    else:
        DEVICE = torch.device('cuda:' + str(args.cuda) if torch.cuda.is_available() else 'cpu')
        setattr(args, 'rank', 0)
        distributed = False

    # Only rank 0 (or single GPU) prints dataset info.
    if (distributed and dist.get_rank() == 0) or not distributed:
        print('Dataset:', args.dataset)
    
    all_errors = []
    
    for seed_idx in range(3):
        if (distributed and dist.get_rank() == 0) or not distributed:
            print(f"Training for seed {seed_idx}")
        # Choose a random seed in the range for each iteration.
        args.seed = np.random.randint(seed_idx * 20, (seed_idx + 1) * 20)
        set_seed(args.seed)
        
        # Get list of domains to iterate over.
        domain_list = set_domain(args)
        seed_errors = []  # To store error for each domain for the current seed
        
        for dom in domain_list:
            setattr(args, 'target_domain', dom)
            error = main_SSL_LE(args, DEVICE)
            if error is not None:
                seed_errors.append(error)
            if distributed:
                dist.barrier()  # Synchronize processes between domain evaluations.
        
        if seed_errors:
            seed_errors = np.array(seed_errors)
            mean_error = np.mean(seed_errors, axis=0)
            std_error = np.std(seed_errors, axis=0)
            if (distributed and dist.get_rank() == 0) or not distributed:
                print(f"Seed {seed_idx}: Mean Errors: M1 = {mean_error[0]:.3f}, M2 = {mean_error[1]:.3f}, M3 = {mean_error[2]:.4f}")
            all_errors.append(mean_error)
    
    if all_errors:
        all_errors = np.array(all_errors)
        overall_mean = np.mean(all_errors, axis=0)
        overall_std = np.std(all_errors, axis=0)
        if (distributed and dist.get_rank() == 0) or not distributed:
            print("Overall Results:")
            print(f"M1 = {overall_mean[0]:.3f}, M2 = {overall_mean[1]:.3f}, M3 = {overall_mean[2]:.4f}")
            print(f"S1 = {overall_std[0]:.3f}, S2 = {overall_std[1]:.3f}, S3 = {overall_std[2]:.4f}")
    
    if distributed:
        dist.destroy_process_group()

if __name__ == '__main__':
    main()