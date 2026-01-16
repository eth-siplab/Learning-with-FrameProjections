import os
import numpy as np
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torchvision import transforms
import torch
import scipy.io
import pickle as cp
from data_preprocess.base_loader import base_loader, base_loader_isoalign
from data_preprocess.augmentations import gen_aug
from data_preprocess.data_preprocess_utils import get_dataset_path
from utils import WaveletTransform, FourierTransform


def load_domain_data(domain_idx):
    str_folder = get_dataset_path('IEEE_Small') + '/'
    data_all = scipy.io.loadmat(str_folder + 'IEEE_Small.mat') 
    data = data_all['whole_dataset']
    domain_idx = int(domain_idx)
    X = data[domain_idx,0]
    y = np.squeeze(data[domain_idx, 1]) - 1
    d = np.full(y.shape, domain_idx, dtype=int)
    return X, y

class data_loader_ieeesmall(base_loader):
    def __init__(self, samples, labels, args):
        super(data_loader_ieeesmall, self).__init__(samples, labels, args)

    def __getitem__(self, index):
        sample, target = self.samples[index], self.labels[index]
        return torch.tensor(sample, device=self.args.cuda).float().unsqueeze(0), torch.tensor(target.item(),device=self.args.cuda).float()

class data_loader_ieeesmall_isoalign(base_loader_isoalign):
    def __init__(self, samples, labels, specs, FT, args):
        super(data_loader_ieeesmall_isoalign, self).__init__(samples, labels, specs, FT, args)

    def __getitem__(self, index):
        sample, target, spec, FT = self.samples[index], self.labels[index], self.specs[index], self.FT[index]
        return (torch.tensor(sample, device=self.args.cuda).float().unsqueeze(0), torch.tensor(target.item(), device=self.args.cuda).float(), 
                spec.clone().float().to(self.args.cuda), FT.to(self.args.cuda).unsqueeze(0)
                )

def prep_domains_ieeesmall_subject_large(args):
    source_domain_list = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11']
    source_domain_list.remove(str(args.target_domain))
    
    # source domain data prep
    xtrain, xbpms, d_win_all = np.array([]), np.array([]), np.array([])
    for source_domain in source_domain_list:
        #print('source_domain:', source_domain)
        x, y = load_domain_data(source_domain)

        x = x.reshape((-1, x.shape[-1]))

        xtrain = np.concatenate((xtrain, x), axis=0) if xtrain.size else x
        xbpms = np.concatenate((xbpms, y), axis=0) if xbpms.size else y

    if args.framework == 'isoalign':
        # Instantiate and compute the wavelet transform
        wavelet_transform = WaveletTransform(wavelet='cmor1-1', fs=25)
        cwt_result, spect_freq, spect_time = wavelet_transform.compute_cwt(xtrain)

        args.spect_freq, args.spect_time = spect_freq, spect_time

        FT_transform = FourierTransform(fs=25)
        FT_result = FT_transform.compute_FT(xtrain)
        data_set = data_loader_ieeesmall_isoalign(xtrain, xbpms, cwt_result, FT_result, args)
    else:
        data_set = data_loader_ieeesmall(xtrain, xbpms, args)
        # source_loader = DataLoader(data_set, batch_size=args.batch_size, shuffle=False)

    x, y = load_domain_data(str(args.target_domain))

    x = x.reshape((-1, x.shape[-1]))
    data_set_test = data_loader_ieeesmall(x, y, args)
    # target_loader = DataLoader(data_set, batch_size=512, shuffle=False)  # For testing keep the batch size as 512

    return data_set, None, data_set_test

def prep_domains_ieeesmall_subject(args):
    source_domain_list = ['0', '1', '2', '3', '4']
    if str(args.target_domain) in source_domain_list: source_domain_list.remove(str(args.target_domain))
    
    # source domain data prep
    xtrain, xbpms = np.array([]), np.array([])
    for source_domain in source_domain_list:
        #print('source_domain:', source_domain)
        x, y = load_domain_data(source_domain)
        x = x.reshape((-1, x.shape[-1]))

        xtrain = np.concatenate((xtrain, x), axis=0) if xtrain.size else x
        xbpms = np.concatenate((xbpms, y), axis=0) if xbpms.size else y

    data_set = data_loader_ieeesmall(xtrain, xbpms, args)
    source_loader = DataLoader(data_set, batch_size=args.batch_size, shuffle=False)

    x, y = load_domain_data(str(args.target_domain))

    x = x.reshape((-1, x.shape[-1]))

    data_set = data_loader_ieeesmall(x, y, args)
    target_loader = DataLoader(data_set, batch_size=512, shuffle=False)  # For testing keep the batch size as 512

    return source_loader, None, target_loader

def prep_domains_ieeesmall_subject_sp(args):
    source_domain_list = ['0', '1', '2', '3', '4']
    # source_domain_list = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11']
    if str(args.target_domain) in source_domain_list: source_domain_list.remove(str(args.target_domain))
    
    # source domain data prep
    xtrain, xbpms = np.array([]), np.array([])
    for source_domain in source_domain_list:
        #print('source_domain:', source_domain)
        x, y = load_domain_data(source_domain)

        x = x.reshape((-1, x.shape[-1]))

        xtrain = np.concatenate((xtrain, x), axis=0) if xtrain.size else x
        xbpms = np.concatenate((xbpms, y), axis=0) if xbpms.size else y

    ###################################################### split the data into training and fine-tuning sets
    # Assuming xtrain and xbpms are your data tensors
    xtrain_shape = xtrain.shape[0]

    # Calculate the number of samples for fine-tuning set (10%)
    fine_tuning_size = int(0.10 * xtrain_shape)

    # Generate random indices for the fine-tuning set
    indices = np.arange(xtrain_shape)
    np.random.shuffle(indices)
    fine_tuning_indices = indices[:fine_tuning_size]
    training_indices = indices[fine_tuning_size:]
    # Split the data into training and fine-tuning sets
    xtrain_fine_tuning = xtrain[fine_tuning_indices]
    xbpms_fine_tuning = xbpms[fine_tuning_indices]
    # 
    xtrain_training = xtrain[training_indices]
    xbpms_training = xbpms[training_indices]    
    #######################################################
    data_set_val = data_loader_ieeesmall(xtrain_fine_tuning, xbpms_fine_tuning, args)
    val_loader = DataLoader(data_set_val, batch_size=args.batch_size, shuffle=False)
    #
    data_set_train = data_loader_ieeesmall(xtrain_training, xbpms_training, args)
    source_loader = DataLoader(data_set_train, batch_size=args.batch_size, shuffle=False)
    source_loaders = [source_loader]

    # Target domain data prep
    x, y = load_domain_data(str(args.target_domain))

    x = x.reshape((-1, x.shape[-1]))

    data_set = data_loader_ieeesmall(x, y, args)
    target_loader = DataLoader(data_set, batch_size=512, shuffle=False)  # For testing keep the batch size as 512

    return source_loaders, val_loader, target_loader


def prep_ieee_small(args):
    if args.cases == 'subject_large' or args.cases == 'subject_large_ssl_fn':
        return prep_domains_ieeesmall_subject_large(args)
    elif args.cases == 'subject_val':
        return prep_domains_ieeesmall_subject_sp(args)
    elif args.cases == 'subject':
        return prep_domains_ieeesmall_subject(args)
    elif args.cases == '':
        pass
    else:
        return 'Error! Unknown args.cases!\n'