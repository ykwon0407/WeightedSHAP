'''
Reference: https://github.com/iancovert/fastshap/blob/main/fastshap/utils.py
'''
import torch
import torch.nn as nn
import numpy as np
import itertools
from torch.utils.data import Dataset

class MaskLayer1d(nn.Module):
    '''
    Masking for 1d inputs.

    Args:
      append: whether to append the mask along channels dim.
      value: replacement value for held out features.
    '''
    def __init__(self, append=True, value=0):
        super().__init__()
        self.append = append
        self.value = value

    def forward(self, input_tuple):
        x, S = input_tuple
        x = x * S + self.value * (1 - S)
        if self.append:
            x = torch.cat((x, S), dim=1)
        return x

class MaskLayer2d(nn.Module):
    '''
    Masking for 2d inputs.

    Args:
      append: whether to append the mask along channels dim.
      value: replacement value for held out features.
    '''
    def __init__(self, append=True, value=0):
        super().__init__()
        self.append = append
        self.value = value

    def forward(self, input_tuple):
        '''
        Apply mask to input.

        Args:
          input_tuple: tuple of input x and mask S.
        '''
        x, S = input_tuple
        x = x * S + self.value * (1 - S)
        if self.append:
            x = torch.cat((x, S), dim=1)
        return x

class KLDivLoss(nn.Module):
    '''
    KL divergence loss that applies log softmax operation to predictions.
    Args:
      reduction: how to reduce loss value (e.g., 'batchmean').
      log_target: whether the target is expected as a log probabilities (or as
        probabilities).
    '''

    def __init__(self, reduction='batchmean', log_target=False):
        super().__init__()
        self.kld = nn.KLDivLoss(reduction=reduction, log_target=log_target)

    def forward(self, pred, target):
        '''
        Evaluate loss.
        Args:
          pred:
          target:
        '''
        return self.kld(pred.log_softmax(dim=1), target)

class MSELoss(nn.Module):
    '''
    MSE loss.
    Args:
      reduction: how to reduce loss value (e.g., 'batchmean').
      log_target: whether the target is expected as a log probabilities (or as
        probabilities).
    '''

    def __init__(self, reduction='mean'):
        super().__init__()
        self.mseloss = nn.MSELoss(reduction=reduction)

    def forward(self, pred, target):
        '''
        Evaluate loss.
        Args:
          pred:
          target:
        '''
        return self.mseloss(pred, target)        

class UniformSampler:
    '''
    For sampling player subsets with cardinality chosen uniformly at random.
    Args:
      num_players: number of players.
    '''

    def __init__(self, num_players):
        self.num_players = num_players

    def sample(self, batch_size):
        '''
        Generate sample.
        Args:
          batch_size:
        '''
        S = torch.ones(batch_size, self.num_players, dtype=torch.float32)
        num_included = (torch.rand(batch_size) * (self.num_players + 1)).int()
        # TODO ideally avoid for loops
        # TODO ideally pass buffer to assign samples in place
        for i in range(batch_size):
            S[i, num_included[i]:] = 0
            S[i] = S[i, torch.randperm(self.num_players)]

        return S

class DatasetRepeat(Dataset):
    '''
    A wrapper around multiple datasets that allows repeated elements when the
    dataset sizes don't match. The number of elements is the maximum dataset
    size, and all datasets must be broadcastable to the same size.
    Args:
      datasets: list of dataset objects.
    '''

    def __init__(self, datasets):
        # Get maximum number of elements.
        assert np.all([isinstance(dset, Dataset) for dset in datasets])
        items = [len(dset) for dset in datasets]
        num_items = np.max(items)

        # Ensure all datasets align.
        # assert np.all([num_items % num == 0 for num in items])
        self.dsets = datasets
        self.num_items = num_items
        self.items = items

    def __getitem__(self, index):
        assert 0 <= index < self.num_items
        return_items = [dset[index % num] for dset, num in
                        zip(self.dsets, self.items)]
        return tuple(itertools.chain(*return_items))

    def __len__(self):
        return self.num_items        

class DatasetInputOnly(Dataset):
    '''
    A wrapper around a dataset object to ensure that only the first element is
    returned.
    Args:
      dataset: dataset object.
    '''

    def __init__(self, dataset):
        assert isinstance(dataset, Dataset)
        self.dataset = dataset

    def __getitem__(self, index):
        return (self.dataset[index][0],)

    def __len__(self):
        return len(self.dataset)

                