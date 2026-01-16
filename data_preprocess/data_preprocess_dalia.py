'''
Data Pre-processing on dalia dataset.

'''

import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
import scipy.io
import pickle as cp
from data_preprocess.base_loader import base_loader, base_loader_isoalign
from data_preprocess.data_preprocess_utils import get_dataset_path
from utils import WaveletTransform, FourierTransform


def load_domain_data(domain_idx):
    str_folder = get_dataset_path('DaLia') + '/'
    file = open(str_folder + 'Dalia_data.pkl', 'rb')
    data = cp.load(file)
    data = data['whole_dataset']
    # np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning) 
    data = np.asarray(data, dtype=object)
    domain_idx = int(domain_idx)
    X = data[domain_idx,0]
    y = np.squeeze(data[domain_idx,1]) - 1
    d = np.full(y.shape, domain_idx, dtype=int)
    return X, y, d

class data_loader_dalia(base_loader):
    def __init__(self, samples, labels, args):
        super(data_loader_dalia, self).__init__(samples, labels, args)

    def __getitem__(self, index):
        sample, target = self.samples[index], self.labels[index]
        return torch.tensor(sample, device=self.args.cuda).float().unsqueeze(0), torch.tensor(target.item(),device=self.args.cuda).float()

class data_loader_dalia_isoalign(base_loader_isoalign):
    def __init__(self, samples, labels, specs, FT, args):
        super(data_loader_dalia_isoalign, self).__init__(samples, labels, specs, FT, args)

    def __getitem__(self, index):
        sample, target, spec, FT = self.samples[index], self.labels[index], self.specs[index], self.FT[index]
        return (torch.tensor(sample, device=self.args.cuda).float().unsqueeze(0), torch.tensor(target.item(),device=self.args.cuda).float(),
                spec.clone().float().to(self.args.cuda), self.FT[index].to(self.args.cuda).unsqueeze(0)
                )

def prep_domains_dalia_subject(args):
    source_domain_list = ['0','1','2', '3','4']
    
    source_domain_list.remove(str(args.target_domain))

    # source domain data prep
    x_win_all, y_win_all, d_win_all = np.array([]), np.array([]), np.array([])
    for source_domain in source_domain_list:
        x, y, d = load_domain_data(source_domain)
        x = x.reshape((-1, x.shape[-1]))

        x_win_all = np.concatenate((x_win_all, x), axis=0) if x_win_all.size else x
        y_win_all = np.concatenate((y_win_all, y), axis=0) if y_win_all.size else y
        d_win_all = np.concatenate((d_win_all, d), axis=0) if d_win_all.size else d

    unique_y, counts_y = np.unique(y_win_all, return_counts=True)

    x_win_all = x_win_all.squeeze()

    data_set = data_loader_dalia(x_win_all, y_win_all, args)
    source_loader = DataLoader(data_set, batch_size=args.batch_size, shuffle=False, drop_last=True)

    x, y, d = load_domain_data(args.target_domain)

    x = x.reshape((-1, x.shape[-1]))

    x = x.squeeze()

    data_set = data_loader_dalia(x, y, args)
    target_loader = DataLoader(data_set, batch_size=args.batch_size, shuffle=False)

    return source_loader, None, target_loader

def prep_domains_dalia_subject_large(args):
    cache_dir = "cache"
    os.makedirs(cache_dir, exist_ok=True)
    
    # Cache file for processed (transformed) source data for IsoAlign.
    processed_cache_file = os.path.join(cache_dir, f"dalia_processed_source.npz")
    
    if args.framework == 'isoalign':
        if os.path.exists(processed_cache_file):
            cached = np.load(processed_cache_file, allow_pickle=True)
            x_all = cached['x']
            y_all = cached['y']
            d_all = cached['d']
            cwt_result = cached['cwt_result']
            spect_freq = cached['spect_freq']
            spect_time = cached['spect_time']
            FT_result = cached['FT_result']
        else:
            print("Processed cache not found. Processing raw data and computing transforms...")
            subject_list = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 
                            '10', '11', '12', '13', '14']
            x_all, y_all, d_all = np.array([]), np.array([]), np.array([])
            for subject in subject_list:
                x, y, d = load_domain_data(subject)

                x = x.reshape((-1, x.shape[-1]))
                x_all = np.concatenate((x_all, x), axis=0) if x_all.size else x
                y_all = np.concatenate((y_all, y), axis=0) if y_all.size else y
                d_all = np.concatenate((d_all, d), axis=0) if d_all.size else d

            # Compute the wavelet transform.
            wavelet_transform = WaveletTransform(wavelet='cmor1-1', fs=25)
            cwt_result, spect_freq, spect_time = wavelet_transform.compute_cwt(x_all)

            # Compute the Fourier transform.
            FT_transform = FourierTransform(fs=25)
            FT_result = FT_transform.compute_FT(x_all)
            # Cache all processed data.
            np.savez_compressed(processed_cache_file, x=x_all, y=y_all, d=d_all,
                                cwt_result=cwt_result, spect_freq=spect_freq,
                                spect_time=spect_time, FT_result=FT_result) # cwt --> [64697, 48, 200, 1], FT --> [64697, 101], x_all --> [64697, 200]

        # Exclude target domain from the data.
        target_domain_str = str(args.target_domain)
        mask = np.array([str(dom) != target_domain_str for dom in d_all])
        x_win_all, y_win_all, d_win_all = x_all[mask], y_all[mask], d_all[mask]
        cwt_result = cwt_result[mask]  if torch.is_tensor(cwt_result) else torch.from_numpy(cwt_result[mask]).float()
        FT_result = FT_result[mask] if torch.is_tensor(FT_result) else torch.from_numpy(FT_result[mask])
        args.spect_freq, args.spect_time = spect_freq, spect_time
            
        # Build dataset for the IsoAlign framework.
        data_set = data_loader_dalia_isoalign(x_win_all, y_win_all, cwt_result, FT_result, args)
    else:
        subject_list = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 
                        '10', '11', '12', '13', '14']

        x_all, y_all, d_all = np.array([]), np.array([]), np.array([])
        for subject in subject_list:
            x, y, d = load_domain_data(subject)
            x = x.reshape((-1, x.shape[-1]))
            x_all = np.concatenate((x_all, x), axis=0) if x_all.size else x
            y_all = np.concatenate((y_all, y), axis=0) if y_all.size else y
            d_all = np.concatenate((d_all, d), axis=0) if d_all.size else d
        
        # Exclude target domain.
        target_domain_str = str(args.target_domain)
        mask = np.array([str(dom) != target_domain_str for dom in d_all])
        x_win_all = x_all[mask]
        y_win_all = y_all[mask]
        d_win_all = d_all[mask]
        
        data_set = data_loader_dalia(x_win_all, y_win_all, args)
    
    source_loader = DataLoader(data_set, batch_size=args.batch_size, shuffle=False, drop_last=True)
    
    # Process target domain separately.
    x, y, d = load_domain_data(str(args.target_domain))
    x = x.reshape((-1, x.shape[-1]))
    data_set_target = data_loader_dalia(x, y, args)
    target_loader = DataLoader(data_set_target, batch_size=args.batch_size, shuffle=False)
    
    return data_set, None, data_set_target

def prep_domains_dalia_subject_sp(args):
    source_domain_list = ['0', '1', '2', '3', '4']
    
    source_domain_list.remove(str(args.target_domain))

    # source_domain_list = [i for i in range(0, 5)] # --> ablation study
    # target_domain_list = [i for i in range(5, 15)]

    # source domain data prep
    xtrain, xbpms = np.array([]), np.array([])
    for source_domain in source_domain_list:
        x, y, d = load_domain_data(source_domain)

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

    data_set = data_loader_dalia(xtrain_fine_tuning, xbpms_fine_tuning, args)
    source_loader = DataLoader(data_set, batch_size=args.batch_size, shuffle=False, drop_last=True)

    ### 
    data_set_val = data_loader_dalia(xtrain_training, xbpms_training, args)
    val_loader = DataLoader(data_set_val, batch_size=args.batch_size, shuffle=False, drop_last=True)
    val_loader = val_loader   

    # target domain data prep
    x, y, d = load_domain_data(args.target_domain)
    x = x.reshape((-1, x.shape[-1]))

    data_set = data_loader_dalia(x, y, args)
    target_loader = DataLoader(data_set, batch_size=args.batch_size, shuffle=False)

    # x_win_all, y_win_all, d_win_all = np.array([]), np.array([]), np.array([])
    # for target_domain in target_domain_list:
    #     x, y, d = load_domain_data(target_domain)
    #     x = np.transpose(x.reshape((-1, 1, 200, 1)), (0, 2, 1, 3))

    #     x_win_all = np.concatenate((x_win_all, x), axis=0) if x_win_all.size else x
    #     y_win_all = np.concatenate((y_win_all, y), axis=0) if y_win_all.size else y
    #     d_win_all = np.concatenate((d_win_all, d), axis=0) if d_win_all.size else d

    # data_set = data_loader_dalia(x_win_all, y_win_all, d_win_all)
    # target_loader = DataLoader(data_set, batch_size=args.batch_size, shuffle=False)

    return [source_loader], val_loader, target_loader

def prep_dalia(args):
    if args.cases == 'subject':
        return prep_domains_dalia_subject(args)
    elif args.cases == 'subject_large' or args.cases == 'subject_large_ssl_fn':
        return prep_domains_dalia_subject_large(args)
    elif args.cases == 'subject_val':
        return prep_domains_dalia_subject_sp(args)
    elif args.cases == '':
        pass
    else:
        return 'Error! Unknown args.cases!\n'

