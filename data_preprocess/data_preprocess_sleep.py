'''
Data Pre-processing on sleep dataset.

'''

import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
import scipy.io
import pickle as cp
from data_preprocess.base_loader import base_loader, base_loader_isoalign
from data_preprocess.data_preprocess_utils import get_data_root
from utils import WaveletTransform, FourierTransform

def load_domain_data():
    str_folder = get_data_root() + '/'
    data = torch.load(str_folder + 'sleep_combined.pt')
    train = data['train']
    val = data['val']
    test = data['test']
    return train, val, test

class data_loader_sleep(base_loader):
    def __init__(self, samples, labels, args):
        super(data_loader_sleep, self).__init__(samples, labels, args)

    def __getitem__(self, index):
        sample, target = self.samples[index], self.labels[index]
        return torch.tensor(sample, device=self.args.cuda).float(), torch.tensor(target.item(), device=self.args.cuda).float()

class data_loader_sleep_isoalign(base_loader_isoalign):
    def __init__(self, samples, labels, specs, FT, args):
        super(data_loader_sleep_isoalign, self).__init__(samples, labels, specs, FT, args)

    def __getitem__(self, index):
        sample, target, spec, FT = self.samples[index], self.labels[index], self.specs[index], self.FT[index]
        sample = np.squeeze(np.transpose(sample, (1, 2, 0)))
        return (torch.tensor(sample, device=self.args.cuda).float(), torch.tensor(target.item(),device=self.args.cuda).float(),
                spec.to(self.args.cuda), FT.to(self.args.cuda)
                )
        # sample --> (9, 128), spec --> (48, 128, 9), FT --> (9, 65)

def prep_domains_sleep_subject_sp(args):
    train, val, test = load_domain_data()

    data_set = data_loader_sleep(train['samples'], train['labels'], np.zeros(train['labels'].shape))
    source_loader = DataLoader(data_set, batch_size=args.batch_size, shuffle=False, drop_last=True, sampler=None)
    source_loaders = [source_loader]

    # 
    data_set_val = data_loader_sleep(val['samples'], val['labels'], np.zeros(val['labels'].shape))
    val_loader = DataLoader(data_set_val, batch_size=args.batch_size, shuffle=False, drop_last=True, sampler=None)
    val_loader = val_loader   

    # target domain data prep

    data_set_test = data_loader_sleep(test['samples'], test['labels'], np.zeros(test['labels'].shape))
    target_loader = DataLoader(data_set_test, batch_size=args.batch_size, shuffle=False)

    return source_loaders, val_loader, target_loader

def prep_domains_sleep_subject_large(args):
    train, val, test = load_domain_data()

    data_set = data_loader_sleep(train['samples'], train['labels'], args)
    # source_loader = DataLoader(data_set, batch_size=args.batch_size, shuffle=False, drop_last=True, sampler=None)
    # 
    data_set_val = data_loader_sleep(val['samples'], val['labels'], args)
    val_loader = DataLoader(data_set_val, batch_size=args.batch_size, shuffle=False, drop_last=True, sampler=None)
    val_loader = val_loader   

    # target domain data prep

    data_set_test = data_loader_sleep(test['samples'], test['labels'], args)
    target_loader = DataLoader(data_set_test, batch_size=args.batch_size, shuffle=False)

    return data_set, None, target_loader

def prep_sleep(args):
    if args.cases == 'subject_val':
        return prep_domains_sleep_subject_sp(args)
    elif args.cases == 'subject':
        return prep_domains_sleep_subject_sp(args)
    elif args.cases == 'subject_large' or args.cases == 'subject_large_ssl_fn':
        return prep_domains_sleep_subject_large(args)
    elif args.cases == '':
        pass
    else:
        return 'Error! Unknown args.cases!\n'