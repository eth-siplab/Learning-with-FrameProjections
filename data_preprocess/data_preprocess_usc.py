'''
Data Pre-processing on USC dataset.

'''

import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
import scipy.io
import pickle as cp
from data_preprocess.data_preprocess_utils import get_sample_weights, train_test_val_split
from data_preprocess.base_loader import base_loader, base_loader_isoalign
from utils import WaveletTransform, FourierTransform


def load_domain_data(domain_idx):
    str_folder = '/data/usc_data/'
    data_all = scipy.io.loadmat(str_folder + 'usc_data.mat')
    data = data_all['whole_dataset']
    domain_idx = int(domain_idx)
    X = data[domain_idx,0]
    y = np.squeeze(data[domain_idx,1]) - 1
    d = np.full(y.shape, domain_idx, dtype=int)
    return X, y, d

class data_loader_usc(base_loader):
    def __init__(self, samples, labels, args):
        super(data_loader_usc, self).__init__(samples, labels, args)

    def __getitem__(self, index):
        sample, target = self.samples[index], self.labels[index]
        sample = np.squeeze(np.transpose(sample, (1, 2, 0)))
        return torch.tensor(sample, device=self.args.cuda).float(), torch.tensor(target.item(), device=self.args.cuda).float()
        # sample -> (6,100)

class data_loader_usc_isoalign(base_loader_isoalign):
    def __init__(self, samples, labels, specs, FT, args):
        super(data_loader_usc_isoalign, self).__init__(samples, labels, specs, FT, args)

    def __getitem__(self, index):
        sample, target, spec, FT = self.samples[index], self.labels[index], self.specs[index], self.FT[index]
        sample = np.squeeze(np.transpose(sample, (1, 2, 0)))
        return (torch.tensor(sample, device=self.args.cuda).float(), torch.tensor(target.item(),device=self.args.cuda).float(),
                spec.to(self.args.cuda), FT.to(self.args.cuda)
                )
        # sample --> (6, 100), spec --> (48, 100, 6), FT --> (4, 100) 

def prep_domains_usc_subject(args):
    source_domain_list = ['10','11', '12', '13']
    
    source_domain_list.remove(str(args.target_domain))

    # source domain data prep
    x_win_all, y_win_all, d_win_all = np.array([]), np.array([]), np.array([])
    for source_domain in source_domain_list:
        #print('source_domain:', source_domain)
        x, y, d = load_domain_data(source_domain)

        x = np.transpose(x.reshape((-1, 1, 100, 6)), (0, 2, 1, 3))

        x_win_all = np.concatenate((x_win_all, x), axis=0) if x_win_all.size else x
        y_win_all = np.concatenate((y_win_all, y), axis=0) if y_win_all.size else y
        d_win_all = np.concatenate((d_win_all, d), axis=0) if d_win_all.size else d
    
    unique_y, counts_y = np.unique(y_win_all, return_counts=True)
    weights = 100.0 / torch.Tensor(counts_y)
    weights = weights.double()

    sample_weights = get_sample_weights(y_win_all, weights)

    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights=sample_weights,
                                                             num_samples=len(sample_weights), replacement=True)

    data_set = data_loader_usc(x_win_all, y_win_all, args)
    source_loader = DataLoader(data_set, batch_size=args.batch_size, shuffle=False, drop_last=True, sampler=sampler)

    # target domain data prep
    #print('target_domain:', args.target_domain)
    x, y, d = load_domain_data(args.target_domain)

    x = np.transpose(x.reshape((-1, 1, 100, 6)), (0, 2, 1, 3))

    data_set = data_loader_usc(x, y, args)
    target_loader = DataLoader(data_set, batch_size=args.batch_size, shuffle=False)

    return source_loader, None, target_loader

def prep_domains_usc_subject_val(args):
    source_domain_list = ['10','11', '12', '13']
    source_domain_list.remove(args.target_domain)
    val_subject = np.random.choice(source_domain_list) # Randomly chose a subject for validation
    source_domain_list.remove(val_subject) # Remove it from training set    
    
    # source domain data prep
    x_win_all, y_win_all, d_win_all = np.array([]), np.array([]), np.array([])
    for source_domain in source_domain_list:
        #print('source_domain:', source_domain)
        x, y, d = load_domain_data(source_domain)
        x = np.transpose(x.reshape((-1, 1, 100, 6)), (0, 2, 1, 3))

        x_win_all = np.concatenate((x_win_all, x), axis=0) if x_win_all.size else x
        y_win_all = np.concatenate((y_win_all, y), axis=0) if y_win_all.size else y
        d_win_all = np.concatenate((d_win_all, d), axis=0) if d_win_all.size else d

    unique_y, counts_y = np.unique(y_win_all, return_counts=True)
    weights = 100.0 / torch.Tensor(counts_y)
    weights = weights.double()
    sample_weights = get_sample_weights(y_win_all, weights)

    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

    # x_win_all --> (n_samples, 100, 1, 6)
    data_set = data_loader_usc(x_win_all, y_win_all, args)
    source_loader = DataLoader(data_set, batch_size=args.batch_size, shuffle=False, drop_last=True, sampler=sampler)
    #print('source_loader batch: ', len(source_loader))
    source_loaders = [source_loader]

    # validation data prep
    #print('val_subject:', val_subject)
    x, y, d = load_domain_data(val_subject)
    x = np.transpose(x.reshape((-1, 1, 100, 6)), (0, 2, 1, 3))
    data_set = data_loader_usc(x, y, args)
    valid_loader = DataLoader(data_set, batch_size=args.batch_size, shuffle=False)

    # target domain data prep
    #print('target_domain:', args.target_domain)
    x, y, d = load_domain_data(args.target_domain)

    x = np.transpose(x.reshape((-1, 1, 100, 6)), (0, 2, 1, 3))

    data_set = data_loader_usc(x, y, args)
    target_loader = DataLoader(data_set, batch_size=args.batch_size, shuffle=False)
    return source_loaders, valid_loader, target_loader

def prep_domains_usc_subject_large(args):
    cache_dir = "cache"
    os.makedirs(cache_dir, exist_ok=True)
    
    # Cache file for processed (transformed) source data for IsoAlign.
    processed_cache_file = os.path.join(cache_dir, f"usc_processed_source.npz")

    if args.framework == 'isoalign':
        if os.path.exists(processed_cache_file):
            cached = np.load(processed_cache_file, allow_pickle=True)
            x_all = cached['x']
            y_all = cached['y']
            d_all = cached['d']
            cwt_result = cached['cwt_result']
            spect_freq = 48
            spect_time = 100
            FT_result = cached['FT_result']
        else:
            print("Processed cache not found. Processing raw data and computing transforms...")
            subject_list = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 
                            '10', '11', '12', '13']
            x_all, y_all, d_all = np.array([]), np.array([]), np.array([])
            for subject in subject_list:
                x, y, d = load_domain_data(subject)
                # Reshape x to (num_samples, 128, 1, 9) as required.
                x = np.transpose(x.reshape((-1, 100, 6)), (0, 1, 2))
                x_all = np.concatenate((x_all, x), axis=0) if x_all.size else x
                y_all = np.concatenate((y_all, y), axis=0) if y_all.size else y
                d_all = np.concatenate((d_all, d), axis=0) if d_all.size else d
            
            # Compute the wavelet transform.
            wavelet_transform = WaveletTransform(wavelet='cmor1-1', fs=50)
            cwt_result, spect_freq, spect_time = wavelet_transform.compute_cwt(x_all)

            # Compute the Fourier transform.
            FT_transform = FourierTransform(fs=50)

            FT_result = FT_transform.compute_FT(x_all.transpose(0, 2, 1))
            
            # Cache all processed data.
            np.savez_compressed(processed_cache_file, x=x_all, y=y_all, d=d_all,
                                cwt_result=cwt_result, spect_freq=spect_freq,
                                spect_time=spect_time, FT_result=FT_result)

        # Exclude target domain from the data.
        target_domain_str = str(args.target_domain)
        mask = np.array([str(dom) != target_domain_str for dom in d_all])
        x_win_all, y_win_all, d_win_all = x_all[mask], y_all[mask], d_all[mask]
        cwt_result = torch.from_numpy(cwt_result[mask]).float()
        FT_result = torch.from_numpy(FT_result[mask])
        args.spect_freq, args.spect_time = spect_freq, spect_time
  
        # Build dataset for the IsoAlign framework.
        data_set = data_loader_usc_isoalign(np.expand_dims(x_win_all, 2), y_win_all, cwt_result.permute(0,2,3,1), FT_result, args)
    else:
        subject_list = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 
                        '10', '11', '12', '13']
        x_all, y_all, d_all = np.array([]), np.array([]), np.array([])
        for subject in subject_list:
            x, y, d = load_domain_data(subject)
            x = np.transpose(x.reshape((-1, 1, 100, 6)), (0, 2, 1, 3))
            x_all = np.concatenate((x_all, x), axis=0) if x_all.size else x
            y_all = np.concatenate((y_all, y), axis=0) if y_all.size else y
            d_all = np.concatenate((d_all, d), axis=0) if d_all.size else d
        
        # Exclude target domain.
        target_domain_str = str(args.target_domain)
        mask = np.array([str(dom) != target_domain_str for dom in d_all])
        x_win_all = x_all[mask]
        y_win_all = y_all[mask]
        d_win_all = d_all[mask]
        
        data_set = data_loader_usc(x_win_all, y_win_all, args)
    
    # source_loader = DataLoader(data_set, batch_size=args.batch_size, shuffle=False, drop_last=True)
    
    # Process target domain separately.
    x, y, d = load_domain_data(str(args.target_domain))
    x = np.transpose(x.reshape((-1, 1, 100, 6)), (0, 2, 1, 3))
    data_set_target = data_loader_usc(x, y, args)
    # target_loader = DataLoader(data_set_target, batch_size=args.batch_size, shuffle=False)
    
    return data_set, None, data_set_target

def prep_usc(args, SLIDING_WINDOW_LEN=0, SLIDING_WINDOW_STEP=0):
    if args.cases == 'subject':
        return prep_domains_usc_subject(args)
    elif args.cases == 'subject_large' or args.cases == 'subject_large_ssl_fn':
        return prep_domains_usc_subject_large(args)
    elif args.cases == 'subject_val':
        return prep_domains_usc_subject_val(args)
    elif args.cases == '':
        pass
    else:
        return 'Error! Unknown args.cases!\n'

