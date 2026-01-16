'''
Data Pre-processing on ecg dataset.

'''

import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
import scipy.io
import pickle as cp
from data_preprocess.data_preprocess_utils import get_sample_weights, train_test_val_split, get_dataset_path
from data_preprocess.base_loader import base_loader, base_loader_isoalign
from utils import WaveletTransform, FourierTransform


def load_domain_data(domain_idx):
    str_folder = get_dataset_path('ECG_data') + '/'
    file = open(str_folder + 'ECG_data.pkl', 'rb')
    data = cp.load(file)
    data = data['whole_dataset']
    domain_idx = int(domain_idx)
    X = data[domain_idx][0]
    y = np.squeeze(data[domain_idx][1]) - 1 if np.min(data[domain_idx][1]) != 0 else np.squeeze(data[domain_idx][1])
    d = np.full(y.shape, domain_idx, dtype=int)
    return X, y, d

class data_loader_ecg(base_loader):
    def __init__(self, samples, labels, args):
        super(data_loader_ecg, self).__init__(samples, labels, args)

    def __getitem__(self, index):
        sample, target = self.samples[index], self.labels[index]
        sample = np.squeeze(np.transpose(sample, (1, 2, 0)))
        return torch.tensor(sample, device=self.args.cuda).float(), torch.tensor(target.item(), device=self.args.cuda).float()
        # sample -> (4,1000)

class data_loader_ecg_isoalign(base_loader_isoalign):
    def __init__(self, samples, labels, specs, FT, args):
        super(data_loader_ecg_isoalign, self).__init__(samples, labels, specs, FT, args)

    def __getitem__(self, index):
        sample, target, spec, FT = self.samples[index], self.labels[index], self.specs[index], self.FT[index]
        sample = np.squeeze(np.transpose(sample, (1, 0)))
        return (torch.tensor(sample, device=self.args.cuda).float(), torch.tensor(target.item(), device=self.args.cuda).float(),
                spec.to(self.args.cuda), FT.to(self.args.cuda)
                )
        # sample --> (4, 1000), spec --> (48, 1000, 4), FT --> (4, 501) 

def prep_cache_ecg_isoalign(args):
    cache_dir = "cache"
    os.makedirs(cache_dir, exist_ok=True)

    if args.dataset == 'chapman':
        processed_cache_file = os.path.join(cache_dir, f"ecg_processed_chapman.npz")
        subject_list = ['0','1','2'] # only the large unsupersived part is transformed for pre-training
    elif args.dataset == 'cpsc':
        processed_cache_file = os.path.join(cache_dir, f"ecg_processed_cpsc.npz")
        subject_list = ['0','2','3']
    else:
        raise ValueError("Unknown dataset. Please specify 'chapman' or 'cpsc'.")

    if os.path.exists(processed_cache_file):
        cached = np.load(processed_cache_file, allow_pickle=True)
        x_all = cached['x']
        y_all = cached['y']
        cwt_result = torch.from_numpy(cached['cwt_result'])
        spect_freq = 48
        spect_time = 1000
        FT_result = torch.from_numpy(cached['FT_result'])
        args.spect_freq, args.spect_time = spect_freq, spect_time
        data_set = data_loader_ecg_isoalign(x_all, y_all, cwt_result.permute(0,2,3,1), FT_result, args)
    else:
        print("Processed cache not found. Processing raw data and computing transforms...")
        x_all, y_all, d_all = np.array([]), np.array([]), np.array([])
        for subject in subject_list:
            x, y, d = load_domain_data(subject)
            # Reshape x to (num_samples, 128, 1, 9) as required.
            x_all = np.concatenate((x_all, x), axis=0) if x_all.size else x
            y_all = np.concatenate((y_all, y), axis=0) if y_all.size else y
            d_all = np.concatenate((d_all, d), axis=0) if d_all.size else d
        
        wavelet_transform = WaveletTransform(wavelet='cmor1-1', fs=100)
        cwt_result, spect_freq, spect_time = wavelet_transform.compute_cwt(x_all) # (B,T,C)
        args.spect_freq, args.spect_time = spect_freq, spect_time
        # Compute the Fourier transform.
        FT_transform = FourierTransform(fs=100)

        FT_result = FT_transform.compute_FT(x_all.transpose(0, 2, 1))
        
        # Cache all processed data.
        np.savez_compressed(processed_cache_file, x=x_all, y=y_all, d=d_all,
                            cwt_result=cwt_result, spect_freq=spect_freq,
                            spect_time=spect_time, FT_result=FT_result)
 
        data_set = data_loader_ecg_isoalign(x_all, y_all, np.transpose(cwt_result,(0,2,3,1)), FT_result, args)

    return data_set


def prep_domains_ecg_subject(args): #  0,1 is for CPSC while 2,3 is for Chapman. 0 (CPSC) and 2 (Chapman) are for fine tuning linear layers.
    source_domain_list = ['0','1'] if args.dataset == 'cpsc' else ['2', '3'] 
    
    source_domain_list.remove(str(args.target_domain))
    # source domain data prep
    x_win_all, y_win_all, d_win_all = np.array([]), np.array([]), np.array([])
    for source_domain in source_domain_list:
        x, y, d = load_domain_data(source_domain)

        x = np.transpose(x.reshape((-1, 1, 1000, 4)), (0, 2, 1, 3))

        x_win_all = np.concatenate((x_win_all, x), axis=0) if x_win_all.size else x
        y_win_all = np.concatenate((y_win_all, y), axis=0) if y_win_all.size else y
        d_win_all = np.concatenate((d_win_all, d), axis=0) if d_win_all.size else d

    unique_y, counts_y = np.unique(y_win_all, return_counts=True)
    weights = 100.0 / torch.Tensor(counts_y)
    weights = weights.double()
    sample_weights = get_sample_weights(y_win_all, weights)

    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights=sample_weights,
                                                             num_samples=len(sample_weights), replacement=True)

    data_set = data_loader_ecg(x_win_all, y_win_all, args)
    source_loader = DataLoader(data_set, batch_size=args.batch_size, shuffle=False, drop_last=True, sampler=sampler)

    # target domain data prep
    x, y, d = load_domain_data(args.target_domain)

    x = np.transpose(x.reshape((-1, 1, 1000, 4)), (0, 2, 1, 3))

    data_set = data_loader_ecg(x, y, args)
    target_loader = DataLoader(data_set, batch_size=args.batch_size, shuffle=False)

    return source_loader, None, target_loader

def prep_domains_ecg_subject_large(args): #  0, 1 is for CPSC while 2,3 is for Chapman
    source_domain_list = ['0', '1', '2', '3']
    
    source_domain_list.remove(str(args.target_domain))
    
    # source domain data prep
    x_win_all, y_win_all, d_win_all = np.array([]), np.array([]), np.array([])
    for source_domain in source_domain_list:
        x, y, d = load_domain_data(source_domain)
        x = np.transpose(x.reshape((-1, 1, 1000, 4)), (0, 2, 1, 3))

        x_win_all = np.concatenate((x_win_all, x), axis=0) if x_win_all.size else x
        y_win_all = np.concatenate((y_win_all, y), axis=0) if y_win_all.size else y
        d_win_all = np.concatenate((d_win_all, d), axis=0) if d_win_all.size else d

    unique_y, counts_y = np.unique(y_win_all, return_counts=True)
    weights = 100.0 / torch.Tensor(counts_y)
    weights = weights.double()
    sample_weights = get_sample_weights(y_win_all, weights)

    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

    if args.framework == 'isoalign':
        data_set_train = prep_cache_ecg_isoalign(args)
    else:
        data_set_train = data_loader_ecg(x_win_all, y_win_all, args)
    # source_loader = DataLoader(data_set, batch_size=args.batch_size, shuffle=False, drop_last=True, sampler=None)

    # target domain data prep
    x, y, d = load_domain_data(args.target_domain)

    x = np.transpose(x.reshape((-1, 1, 1000, 4)), (0, 2, 1, 3))

    data_set = data_loader_ecg(x, y, args)
    target_loader = DataLoader(data_set, batch_size=args.batch_size, shuffle=False)
    return data_set_train, None, target_loader

def prep_domains_ecg_sp(args):
    source_domain_list = ['0','1'] if args.target_domain == '1' else ['2', '3']

    x_win_all, y_win_all, d_win_all = np.array([]), np.array([]), np.array([])
    n_train, n_test, split_ratio = [], 0, 0.0

    for source_domain in source_domain_list:
        x_win, y_win, d_win = load_domain_data(source_domain)

        x_win = np.transpose(x_win.reshape((-1, 1, 1000, 4)), (0, 2, 1, 3))

        x_win_all = np.concatenate((x_win_all, x_win), axis=0) if x_win_all.size else x_win
        y_win_all = np.concatenate((y_win_all, y_win), axis=0) if y_win_all.size else y_win
        d_win_all = np.concatenate((d_win_all, d_win), axis=0) if d_win_all.size else d_win
        n_train.append(x_win.shape[0])

    x_win_train, x_win_val, x_win_test, \
    y_win_train, y_win_val, y_win_test, \
    d_win_train, d_win_val, d_win_test = train_test_val_split(x_win_all, y_win_all, d_win_all, split_ratio=args.split_ratio)

    unique_y, counts_y = np.unique(y_win_train, return_counts=True)
    # print('y_train label distribution: ', dict(zip(unique_y, counts_y)))
    weights = 100.0 / torch.Tensor(counts_y)
    # print('weights of sampler: ', weights)
    weights = weights.double()
    sample_weights = get_sample_weights(y_win_train, weights)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

    train_set_r = data_loader_ecg(x_win_train, y_win_train, args)
    train_loader_r = DataLoader(train_set_r, batch_size=args.batch_size, shuffle=False, drop_last=True, sampler=sampler)
    val_set_r = data_loader_ecg(x_win_val, y_win_val, args)
    val_loader_r = DataLoader(val_set_r, batch_size=args.batch_size, shuffle=False)
    test_set_r = data_loader_ecg(x_win_test, y_win_test, args)
    test_loader_r = DataLoader(test_set_r, batch_size=args.batch_size, shuffle=False)

    return [train_loader_r], val_loader_r, test_loader_r

def prep_ecg(args):
    if args.cases == 'subject':
        return prep_domains_ecg_subject(args)
    elif args.cases == 'subject_large' or args.cases == 'subject_large_ssl_fn':
        return prep_domains_ecg_subject_large(args)
    elif args.cases == 'subject_val':
        return prep_domains_ecg_sp(args)
    elif args.cases == '':
        pass
    else:
        return 'Error! Unknown args.cases!\n'