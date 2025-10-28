import numpy as np
from torch.utils.data import Dataset

class base_loader(Dataset):
    def __init__(self, samples, labels, args):
        self.samples = samples
        self.labels = labels
        self.args = args

    def __getitem__(self, index):
        sample, target, args = self.samples[index], self.labels[index], self.args[index]
        return sample, target, args

    def __len__(self):
        return len(self.samples)

class base_loader_isoalign(Dataset):
    def __init__(self, samples, labels, specs, FT, args):
        self.samples = samples
        self.labels = labels
        self.specs = specs
        self.FT = FT
        self.args = args

    def __getitem__(self, index):
        sample, target, specs, FT = self.samples[index], self.labels[index], self.specs[index], self.FT[index]
        return sample, target, specs

    def __len__(self):
        return len(self.samples)


