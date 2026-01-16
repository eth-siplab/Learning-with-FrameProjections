'''
Data Pre-processing on UCIHAR dataset.

'''

import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
import pickle as cp
from data_preprocess.data_preprocess_utils import get_sample_weights, train_test_val_split, train_val_split, get_dataset_path
from data_preprocess.base_loader import base_loader, base_loader_isoalign
from utils import WaveletTransform, FourierTransform

def format_data_x(datafile):
    x_data = None
    for item in datafile:
        item_data = np.loadtxt(item, dtype=float)
        if x_data is None:
            x_data = np.zeros((len(item_data), 1))
        x_data = np.hstack((x_data, item_data))
    x_data = x_data[:, 1:]
    print('x_data.shape:', x_data.shape)
    X = None
    for i in range(len(x_data)):
        row = np.asarray(x_data[i, :])
        row = row.reshape(9, 128).T
        if X is None:
            X = np.zeros((len(x_data), 128, 9))
        X[i] = row
    print('X.shape:', X.shape)
    return X

def format_data_y(datafile):
    data = np.loadtxt(datafile, dtype=np.int) - 1
    return data

def load_domain_data(domain_idx):
    """ to load all the data from the specific domain with index domain_idx
    :param domain_idx: index of a single domain
    :return: X and y data of the entire domain
    """
    data_dir = get_dataset_path('ucihar/')
    saved_filename = 'ucihar_domain_' + domain_idx + '_wd.data' # "wd": with domain label

    if os.path.isfile(data_dir + saved_filename) == True:
        data = np.load(data_dir + saved_filename, allow_pickle=True)
        X = data[0][0]
        y = data[0][1]
        d = data[0][2]
    else:
        if os.path.isdir(data_dir) == False:
            os.makedirs(data_dir)
        str_folder = get_dataset_path('UCI HAR Dataset/')
        INPUT_SIGNAL_TYPES = [
            "body_acc_x_",
            "body_acc_y_",
            "body_acc_z_",
            "body_gyro_x_",
            "body_gyro_y_",
            "body_gyro_z_",
            "total_acc_x_",
            "total_acc_y_",
            "total_acc_z_"
        ]
        str_train_files = [str_folder + 'train/' + 'Inertial Signals/' + item + 'train.txt' for item in INPUT_SIGNAL_TYPES]
        str_test_files = [str_folder + 'test/' + 'Inertial Signals/' + item + 'test.txt' for item in INPUT_SIGNAL_TYPES]
        str_train_y = str_folder + 'train/y_train.txt'
        str_test_y = str_folder + 'test/y_test.txt'

        str_train_id = str_folder + 'train/subject_train.txt'
        str_test_id = str_folder + 'test/subject_test.txt'

        X_train = format_data_x(str_train_files)
        X_test = format_data_x(str_test_files)
        Y_train = format_data_y(str_train_y)
        Y_test = format_data_y(str_test_y)
        id_train = format_data_y(str_train_id) # origin: array([ 0,  2,  4,  5,  6,  7, 10, 13, 14, 15, 16, 18, 20, 21, 22, 24, 25, 26, 27, 28, 29])
        id_test = format_data_y(str_test_id) # origin: array([ 1,  3,  8,  9, 11, 12, 17, 19, 23])

        X_all = np.concatenate((X_train, X_test), axis=0)
        y_all = np.concatenate((Y_train, Y_test), axis=0)
        id_all = np.concatenate((id_train, id_test), axis=0)

        print('\nProcessing domain {0} files...\n'.format(domain_idx))
        target_idx = np.where(id_all == int(domain_idx))
        X = X_all[target_idx]
        y = y_all[target_idx]
        d = np.full(y.shape, int(domain_idx), dtype=int)
        print('\nProcessing domain {0} files | X: {1} y: {2} d:{3} \n'.format(domain_idx, X.shape, y.shape, d.shape))

        obj = [(X, y, d)]
        f = open(os.path.join(data_dir, saved_filename), 'wb')
        cp.dump(obj, f, protocol=cp.HIGHEST_PROTOCOL)
        f.close()
    return X, y, d

class data_loader_ucihar(base_loader):
    def __init__(self, samples, labels, args):
        super(data_loader_ucihar, self).__init__(samples, labels, args)

    def __getitem__(self, index):
        sample, target = self.samples[index], self.labels[index]
        sample = np.squeeze(np.transpose(sample, (1, 2, 0)))
        return torch.tensor(sample, device=self.args.cuda).float(), torch.tensor(target.item(), device=self.args.cuda).float()

class data_loader_ucihar_isoalign(base_loader_isoalign):
    def __init__(self, samples, labels, specs, FT, args):
        super(data_loader_ucihar_isoalign, self).__init__(samples, labels, specs, FT, args)

    def __getitem__(self, index):
        sample, target, spec, FT = self.samples[index], self.labels[index], self.specs[index], self.FT[index]
        sample = np.squeeze(np.transpose(sample, (1, 2, 0)))
        return (torch.tensor(sample, device=self.args.cuda).float(), torch.tensor(target.item(),device=self.args.cuda).float(),
                spec.to(self.args.cuda), FT.to(self.args.cuda)
                )
        # sample --> (9, 128), spec --> (48, 128, 9), FT --> (9, 65)

def prep_domains_ucihar_subject(args):
    source_domain_list = ['0', '1', '2', '3', '4']
    
    source_domain_list.remove(str(args.target_domain))

    # source domain data prep
    x_win_all, y_win_all, d_win_all = np.array([]), np.array([]), np.array([])
    for source_domain in source_domain_list:
        #print('source_domain:', source_domain)
        x, y, d = load_domain_data(source_domain)

        # n_channel should be 9, H: 1, W:128
        x = np.transpose(x.reshape((-1, 1, 128, 9)), (0, 2, 1, 3))

        x_win_all = np.concatenate((x_win_all, x), axis=0) if x_win_all.size else x
        y_win_all = np.concatenate((y_win_all, y), axis=0) if y_win_all.size else y
        d_win_all = np.concatenate((d_win_all, d), axis=0) if d_win_all.size else d

    x_win_train, x_win_val, \
    y_win_train, y_win_val, \
    d_win_train, d_win_val = train_val_split(x_win_all, y_win_all, d_win_all, split_ratio=0.1)          

    unique_y, counts_y = np.unique(y_win_train, return_counts=True)
    #print('y_train label distribution: ', dict(zip(unique_y, counts_y)))
    weights = 100.0 / torch.Tensor(counts_y)
    #print('weights of sampler: ', weights)
    weights = weights.double()

    sample_weights = get_sample_weights(y_win_train, weights)

    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights=sample_weights,
                                                             num_samples=len(sample_weights), replacement=True)

    data_set = data_loader_ucihar(x_win_train, y_win_train, args)

    source_loader = DataLoader(data_set, batch_size=args.batch_size, shuffle=False, drop_last=True, sampler=sampler)

    ### validation domain data prep
    data_set_val = data_loader_ucihar(x_win_val, y_win_val, args)
    val_loader = DataLoader(data_set_val, batch_size=args.batch_size, shuffle=False, sampler=None)    

    # target domain data prep
    x, y, d = load_domain_data(str(args.target_domain))

    x = np.transpose(x.reshape((-1, 1, 128, 9)), (0, 2, 1, 3))

    data_set = data_loader_ucihar(x, y, args)
    target_loader = DataLoader(data_set, batch_size=args.batch_size, shuffle=False)

    return source_loader, val_loader, target_loader

def prep_domains_ucihar_subject_large(args):
    cache_dir = "cache"
    os.makedirs(cache_dir, exist_ok=True)
    
    # Cache file for processed (transformed) source data for IsoAlign.
    processed_cache_file = os.path.join(cache_dir, f"ucihar_processed_source.npz")
    
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
                            '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', 
                            '20', '21', '22', '23', '24', '25', '26', '27', '28', '29']
            x_all, y_all, d_all = np.array([]), np.array([]), np.array([])
            for subject in subject_list:
                x, y, d = load_domain_data(subject)
                # Reshape x to (num_samples, 128, 1, 9) as required.
                x = np.transpose(x.reshape((-1, 1, 128, 9)), (0, 2, 1, 3))
                x_all = np.concatenate((x_all, x), axis=0) if x_all.size else x
                y_all = np.concatenate((y_all, y), axis=0) if y_all.size else y
                d_all = np.concatenate((d_all, d), axis=0) if d_all.size else d
            
            # Compute the wavelet transform.
            wavelet_transform = WaveletTransform(wavelet='cmor1-1', fs=50)
            cwt_result, spect_freq, spect_time = wavelet_transform.compute_cwt(x_all)

            # Compute the Fourier transform.
            FT_transform = FourierTransform(fs=50)
            FT_result = FT_transform.compute_FT(x_all)
            
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
        data_set = data_loader_ucihar_isoalign(x_win_all, y_win_all, cwt_result, FT_result, args)
    else:
        subject_list = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 
                        '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', 
                        '20', '21', '22', '23', '24', '25', '26', '27', '28', '29']
        x_all, y_all, d_all = np.array([]), np.array([]), np.array([])
        for subject in subject_list:
            x, y, d = load_domain_data(subject)
            x = np.transpose(x.reshape((-1, 1, 128, 9)), (0, 2, 1, 3))
            x_all = np.concatenate((x_all, x), axis=0) if x_all.size else x
            y_all = np.concatenate((y_all, y), axis=0) if y_all.size else y
            d_all = np.concatenate((d_all, d), axis=0) if d_all.size else d
        
        # Exclude target domain.
        target_domain_str = str(args.target_domain)
        mask = np.array([str(dom) != target_domain_str for dom in d_all])
        x_win_all = x_all[mask]
        y_win_all = y_all[mask]
        d_win_all = d_all[mask]
        
        data_set = data_loader_ucihar(x_win_all, y_win_all, args)
    
    # source_loader = DataLoader(data_set, batch_size=args.batch_size, shuffle=False, drop_last=True)
    
    # Process target domain separately.
    x, y, d = load_domain_data(str(args.target_domain))
    x = np.transpose(x.reshape((-1, 1, 128, 9)), (0, 2, 1, 3))
    data_set_target = data_loader_ucihar(x, y, args)
    # target_loader = DataLoader(data_set_target, batch_size=args.batch_size, shuffle=False)
    
    return data_set, None, data_set_target

def prep_domains_ucihar_subject_val(args):
    source_domain_list = ['0', '1', '2', '3', '4']
    source_domain_list = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
    # source_domain_list = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29']
    source_domain_list.remove(args.target_domain) # Remove the target domain from training set
    val_subject = np.random.choice(source_domain_list) # Randomly chose a subject for validation
    source_domain_list.remove(val_subject) # Remove it from training set

    # source domain data prep
    x_win_all, y_win_all, d_win_all = np.array([]), np.array([]), np.array([])
    for source_domain in source_domain_list:
        x, y, d = load_domain_data(source_domain)

        # n_channel should be 9, H: 1, W:128    
        x = np.transpose(x.reshape((-1, 1, 128, 9)), (0, 2, 1, 3))

        x_win_all = np.concatenate((x_win_all, x), axis=0) if x_win_all.size else x
        y_win_all = np.concatenate((y_win_all, y), axis=0) if y_win_all.size else y
        d_win_all = np.concatenate((d_win_all, d), axis=0) if d_win_all.size else d

    unique_y, counts_y = np.unique(y_win_all, return_counts=True)
    #print('y_train label distribution: ', dict(zip(unique_y, counts_y)))
    weights = 100.0 / torch.Tensor(counts_y)
    #print('weights of sampler: ', weights)
    weights = weights.double()

    sample_weights = get_sample_weights(y_win_all, weights)

    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

    # x_win_all --> (n_samples, 128, 1, 9)
    data_set = data_loader_ucihar(x_win_all, y_win_all, args)
    source_loader = DataLoader(data_set, batch_size=args.batch_size, shuffle=False, drop_last=True, sampler=sampler)
    #print('source_loader batch: ', len(source_loader))
    source_loaders = [source_loader]

    # validation data prep
    x, y, d = load_domain_data(val_subject)
    x = np.transpose(x.reshape((-1, 1, 128, 9)), (0, 2, 1, 3))
    data_set = data_loader_ucihar(x, y, args)
    valid_loader = DataLoader(data_set, batch_size=args.batch_size, drop_last=True, shuffle=False)
    # target domain data prep
    #print('target_domain:', args.target_domain)
    x, y, d = load_domain_data(args.target_domain)

    x = np.transpose(x.reshape((-1, 1, 128, 9)), (0, 2, 1, 3))
    #print(" ..after sliding window: inputs {0}, targets {1}".format(x.shape, y.shape))

    data_set = data_loader_ucihar(x, y, args)
    # todo: the batch size can be different for some ttt models, tbc
    target_loader = DataLoader(data_set, batch_size=args.batch_size, drop_last=True, shuffle=False)
    #print('target_loader batch: ', len(target_loader))

    return source_loaders, valid_loader, target_loader

def prep_domains_ucihar_random(args):
    source_domain_list = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29']

    x_win_all, y_win_all, d_win_all = np.array([]), np.array([]), np.array([])
    n_train, n_test, split_ratio = [], 0, 0.0

    for source_domain in source_domain_list:
        # print('source_domain:', source_domain)
        x_win, y_win, d_win = load_domain_data(source_domain)

        # n_channel should be 9, H: 1, W:128
        x_win = np.transpose(x_win.reshape((-1, 1, 128, 9)), (0, 2, 1, 3))
        # print(" ..after sliding window: inputs {0}, targets {1}".format(x_win.shape, y_win.shape))

        x_win_all = np.concatenate((x_win_all, x_win), axis=0) if x_win_all.size else x_win
        y_win_all = np.concatenate((y_win_all, y_win), axis=0) if y_win_all.size else y_win
        d_win_all = np.concatenate((d_win_all, d_win), axis=0) if d_win_all.size else d_win
        n_train.append(x_win.shape[0])

    x_win_train, x_win_val, x_win_test, \
    y_win_train, y_win_val, y_win_test, \
    d_win_train, d_win_val, d_win_test = train_test_val_split(x_win_all, y_win_all, d_win_all, split_ratio=args.split_ratio)

    unique_y, counts_y = np.unique(y_win_train, return_counts=True)
    print('y_train label distribution: ', dict(zip(unique_y, counts_y)))
    weights = 100.0 / torch.Tensor(counts_y)
    print('weights of sampler: ', weights)
    weights = weights.double()
    sample_weights = get_sample_weights(y_win_train, weights)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0, 0, 0, 0, 0, 0, 0, 0, 0), std=(1, 1, 1, 1, 1, 1, 1, 1, 1))
    ])
    train_set_r = data_loader_ucihar(x_win_train, y_win_train, d_win_train, transform)
    train_loader_r = DataLoader(train_set_r, batch_size=args.batch_size, shuffle=False, drop_last=True, sampler=sampler)
    val_set_r = data_loader_ucihar(x_win_val, y_win_val, d_win_val, transform)
    val_loader_r = DataLoader(val_set_r, batch_size=args.batch_size, shuffle=False)
    test_set_r = data_loader_ucihar(x_win_test, y_win_test, d_win_test, transform)
    test_loader_r = DataLoader(test_set_r, batch_size=args.batch_size, shuffle=False)

    return [train_loader_r], val_loader_r, test_loader_r

def prep_ucihar(args):
    if args.cases == 'random':
        return prep_domains_ucihar_random(args)
    elif args.cases == 'subject':
        return prep_domains_ucihar_subject(args)
    elif args.cases == 'subject_large' or args.cases == 'subject_large_ssl_fn':
        return prep_domains_ucihar_subject_large(args)
    elif args.cases == 'subject_val':
        return prep_domains_ucihar_subject_val(args)
    elif args.cases == '':
        pass
    else:
        return 'Error! Unknown args.cases!\n'

