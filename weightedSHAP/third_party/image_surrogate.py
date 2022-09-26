'''
Reference: https://github.com/iancovert/fastshap/blob/main/fastshap/image_surrogate.py
'''
import torch
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, TensorDataset, DataLoader
from torch.utils.data import RandomSampler, BatchSampler
from .image_imputers import ImageImputer
from .utils import UniformSampler, DatasetRepeat
from tqdm.auto import tqdm
from copy import deepcopy

def validate(surrogate, loss_fn, data_loader):
    '''
    Calculate mean validation loss.
    Args:
      loss_fn: loss function.
      data_loader: data loader.
    '''
    with torch.no_grad():
        # Setup.
        device = next(surrogate.surrogate.parameters()).device
        mean_loss = 0
        N = 0

        for x, y, S in data_loader:
            x = x.to(device)
            y = y.to(device)
            S = S.to(device)
            pred = surrogate(x, S)
            loss = loss_fn(pred, y)
            N += len(x)
            mean_loss += len(x) * (loss - mean_loss) / N

    return mean_loss


def generate_labels(dataset, model, batch_size, num_workers):
    '''
    Generate prediction labels for a set of inputs.
    Args:
      dataset: dataset object.
      model: predictive model.
      batch_size: minibatch size.
      num_workers: number of worker threads.
    '''
    with torch.no_grad():
        # Setup.
        preds = []
        device = next(model.parameters()).device
        loader = DataLoader(dataset, batch_size=batch_size, pin_memory=True,
                            num_workers=num_workers)

        for (x,) in loader:
            pred = model(x.to(device)).cpu()
            preds.append(pred)

    return torch.cat(preds)


class ImageSurrogate(ImageImputer):
    '''
    Wrapper around image surrogate model.
    Args:
      surrogate: surrogate model (torch.nn.Module).
      width: image width.
      height: image height.
      superpixel_size: superpixel width/height (int).
    '''

    def __init__(self, surrogate, width, height, superpixel_size):
        # Initialize for coalition resizing, number of players.
        super().__init__(width, height, superpixel_size)

        # Store surrogate model.
        self.surrogate = surrogate

    def train(self,
              train_data,
              val_data,
              batch_size,
              max_epochs,
              loss_fn,
              validation_samples=1,
              validation_batch_size=None,
              lr=1e-3,
              min_lr=1e-5,
              lr_factor=0.5,
              lookback=5,
              training_seed=None,
              validation_seed=None,
              num_workers=0,
              bar=False,
              verbose=False):
        '''
        Train surrogate model.
        Args:
          train_data: training data with inputs and the original model's
            predictions (np.ndarray tuple, torch.Tensor tuple,
            torch.utils.data.Dataset).
          val_data: validation data with inputs and the original model's
            predictions (np.ndarray tuple, torch.Tensor tuple,
            torch.utils.data.Dataset).
          batch_size: minibatch size.
          max_epochs: max number of training epochs.
          loss_fn: loss function (e.g., fastshap.KLDivLoss)
          validation_samples: number of samples per validation example.
          validation_batch_size: validation minibatch size.
          lr: initial learning rate.
          min_lr: minimum learning rate.
          lr_factor: learning rate decrease factor.
          lookback: lookback window for early stopping.
          training_seed: random seed for training.
          validation_seed: random seed for generating validation data.
          num_workers: number of worker threads in data loader.
          bar: whether to show progress bar.
          verbose: verbosity.
        '''
        # Set up train dataset.
        if isinstance(train_data, tuple):
            x_train, y_train = train_data
            if isinstance(x_train, np.ndarray):
                x_train = torch.tensor(x_train, dtype=torch.float32)
                y_train = torch.tensor(y_train, dtype=torch.float32)
            train_set = TensorDataset(x_train, y_train)
        elif isinstance(train_data, Dataset):
            train_set = train_data
        else:
            raise ValueError('train_data must be either tuple of tensors or a '
                             'PyTorch Dataset')

        # Set up train data loader.
        random_sampler = RandomSampler(
            train_set, replacement=True,
            num_samples=int(np.ceil(len(train_set) / batch_size))*batch_size)
        batch_sampler = BatchSampler(
            random_sampler, batch_size=batch_size, drop_last=True)
        train_loader = DataLoader(train_set, batch_sampler=batch_sampler,
                                  pin_memory=True, num_workers=num_workers)

        # Set up validation dataset.
        sampler = UniformSampler(self.num_players)
        if validation_seed is not None:
            torch.manual_seed(validation_seed)
        S_val = sampler.sample(len(val_data) * validation_samples)

        if isinstance(val_data, tuple):
            x_val, y_val = val_data
            if isinstance(x_val, np.ndarray):
                x_val = torch.tensor(x_val, dtype=torch.float32)
                y_val = torch.tensor(y_val, dtype=torch.float32)
            x_val_repeat = x_val.repeat(validation_samples, 1, 1, 1)
            y_val_repeat = y_val.repeat(validation_samples, 1)
            val_set = TensorDataset(x_val_repeat, y_val_repeat, S_val)
        elif isinstance(val_data, Dataset):
            val_set = DatasetRepeat([val_data, TensorDataset(S_val)])
        else:
            raise ValueError('val_data must be either tuple of tensors or a '
                             'PyTorch Dataset')

        if validation_batch_size is None:
            validation_batch_size = batch_size
        val_loader = DataLoader(val_set, batch_size=validation_batch_size,
                                pin_memory=True, num_workers=num_workers)

        # Setup for training.
        surrogate = self.surrogate
        device = next(surrogate.parameters()).device
        optimizer = optim.Adam(surrogate.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=lr_factor, patience=lookback // 2, min_lr=min_lr,
            verbose=verbose)
        best_loss = validate(self, loss_fn, val_loader).item()
        best_epoch = 0
        best_model = deepcopy(surrogate)
        loss_list = [best_loss]
        if training_seed is not None:
            torch.manual_seed(training_seed)

        for epoch in range(max_epochs):
            # Batch iterable.
            if bar:
                batch_iter = tqdm(train_loader, desc='Training epoch')
            else:
                batch_iter = train_loader

            for x, y in batch_iter:
                # Prepare data.
                x = x.to(device)
                y = y.to(device)

                # Generate subsets.
                S = sampler.sample(batch_size).to(device=device)

                # Make predictions.
                pred = self.__call__(x, S)
                loss = loss_fn(pred, y)

                # Optimizer step.
                loss.backward()
                optimizer.step()
                surrogate.zero_grad()

            # Evaluate validation loss.
            self.surrogate.eval()
            val_loss = validate(self, loss_fn, val_loader).item()
            self.surrogate.train()

            # Print progress.
            if verbose:
                print('----- Epoch = {} -----'.format(epoch + 1))
                print('Val loss = {:.4f}'.format(val_loss))
                print('')
            scheduler.step(val_loss)
            loss_list.append(val_loss)

            # Check if best model.
            if val_loss < best_loss:
                best_loss = val_loss
                best_model = deepcopy(surrogate)
                best_epoch = epoch
                if verbose:
                    print('New best epoch, loss = {:.4f}'.format(val_loss))
                    print('')
            elif epoch - best_epoch == lookback:
                if verbose:
                    print('Stopping early')
                break

        # Clean up.
        for param, best_param in zip(surrogate.parameters(),
                                     best_model.parameters()):
            param.data = best_param.data
        self.loss_list = loss_list
        self.surrogate.eval()

    def train_original_model(self,
                             train_data,
                             val_data,
                             original_model,
                             batch_size,
                             max_epochs,
                             loss_fn,
                             validation_samples=1,
                             validation_batch_size=None,
                             lr=1e-3,
                             min_lr=1e-5,
                             lr_factor=0.5,
                             lookback=5,
                             training_seed=None,
                             validation_seed=None,
                             num_workers=0,
                             bar=False,
                             verbose=False):
        '''
        Train surrogate model with labels provided by the original model. This
        approach is designed for when data augmentations make the data loader
        non-deterministic, and labels (the original model's predictions) cannot
        be generated prior to training.
        Args:
          train_data: training data with inputs only (np.ndarray, torch.Tensor,
            torch.utils.data.Dataset).
          val_data: validation data with inputs only (np.ndarray, torch.Tensor,
            torch.utils.data.Dataset).
          original_model: original predictive model (e.g., torch.nn.Module).
          batch_size: minibatch size.
          max_epochs: max number of training epochs.
          loss_fn: loss function (e.g., fastshap.KLDivLoss)
          validation_samples: number of samples per validation example.
          validation_batch_size: validation minibatch size.
          lr: initial learning rate.
          min_lr: minimum learning rate.
          lr_factor: learning rate decrease factor.
          lookback: lookback window for early stopping.
          training_seed: random seed for training.
          validation_seed: random seed for generating validation data.
          num_workers: number of worker threads in data loader.
          bar: whether to show progress bar.
          verbose: verbosity.
        '''
        # Set up train dataset.
        if isinstance(train_data, np.ndarray):
            train_data = torch.tensor(train_data, dtype=torch.float32)

        if isinstance(train_data, torch.Tensor):
            train_set = TensorDataset(train_data)
        elif isinstance(train_data, Dataset):
            train_set = train_data
        else:
            raise ValueError('train_data must be either tensor or a '
                             'PyTorch Dataset')

        # Set up train data loader.
        random_sampler = RandomSampler(
            train_set, replacement=True,
            num_samples=int(np.ceil(len(train_set) / batch_size))*batch_size)
        batch_sampler = BatchSampler(
            random_sampler, batch_size=batch_size, drop_last=True)
        train_loader = DataLoader(train_set, batch_sampler=batch_sampler,
                                  pin_memory=True, num_workers=num_workers)

        # Set up validation dataset.
        sampler = UniformSampler(self.num_players)
        if validation_seed is not None:
            torch.manual_seed(validation_seed)
        S_val = sampler.sample(len(val_data) * validation_samples)
        if validation_batch_size is None:
            validation_batch_size = batch_size

        if isinstance(val_data, np.ndarray):
            val_data = torch.tensor(val_data, dtype=torch.float32)

        if isinstance(val_data, torch.Tensor):
            # Generate validation labels.
            y_val = generate_labels(TensorDataset(val_data), original_model,
                                    validation_batch_size, num_workers)
            y_val_repeat = y_val.repeat(
                validation_samples, *[1 for _ in y_val.shape[1:]])

            # Create dataset.
            val_data_repeat = val_data.repeat(validation_samples, 1, 1, 1)
            val_set = TensorDataset(val_data_repeat, y_val_repeat, S_val)
        elif isinstance(val_data, Dataset):
            # Generate validation labels.
            y_val = generate_labels(val_data, original_model,
                                    validation_batch_size, num_workers)
            y_val_repeat = y_val.repeat(
                validation_samples, *[1 for _ in y_val.shape[1:]])

            # Create dataset.
            val_set = DatasetRepeat(
                [val_data, TensorDataset(y_val_repeat, S_val)])
        else:
            raise ValueError('val_data must be either tuple of tensors or a '
                             'PyTorch Dataset')

        val_loader = DataLoader(val_set, batch_size=validation_batch_size,
                                pin_memory=True, num_workers=num_workers)

        # Setup for training.
        surrogate = self.surrogate
        device = next(surrogate.parameters()).device
        optimizer = optim.Adam(surrogate.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=lr_factor, patience=lookback // 2, min_lr=min_lr,
            verbose=verbose)
        best_loss = validate(self, loss_fn, val_loader).item()
        best_epoch = 0
        best_model = deepcopy(surrogate)
        loss_list = [best_loss]
        if training_seed is not None:
            torch.manual_seed(training_seed)

        for epoch in range(max_epochs):
            # Batch iterable.
            if bar:
                batch_iter = tqdm(train_loader, desc='Training epoch')
            else:
                batch_iter = train_loader

            for (x,) in batch_iter:
                # Prepare data.
                x = x.to(device)

                # Get original model prediction.
                with torch.no_grad():
                    y = original_model(x)

                # Generate subsets.
                S = sampler.sample(batch_size).to(device=device)

                # Make predictions.
                pred = self.__call__(x, S)
                loss = loss_fn(pred, y)

                # Optimizer step.
                loss.backward()
                optimizer.step()
                surrogate.zero_grad()

            # Evaluate validation loss.
            self.surrogate.eval()
            val_loss = validate(self, loss_fn, val_loader).item()
            self.surrogate.train()

            # Print progress.
            if verbose:
                print('----- Epoch = {} -----'.format(epoch + 1))
                print('Val loss = {:.4f}'.format(val_loss))
                print('')
            scheduler.step(val_loss)
            loss_list.append(val_loss)

            # Check if best model.
            if val_loss < best_loss:
                best_loss = val_loss
                best_model = deepcopy(surrogate)
                best_epoch = epoch
                if verbose:
                    print('New best epoch, loss = {:.4f}'.format(val_loss))
                    print('')
            elif epoch - best_epoch == lookback:
                if verbose:
                    print('Stopping early')
                break

        # Clean up.
        for param, best_param in zip(surrogate.parameters(),
                                     best_model.parameters()):
            param.data = best_param.data
        self.loss_list = loss_list
        self.surrogate.eval()

    def __call__(self, x, S):
        '''
        Evaluate surrogate model.
        Args:
          x: input examples.
          S: coalitions.
        '''
        S = self.resize(S)
        return self.surrogate((x, S))


        