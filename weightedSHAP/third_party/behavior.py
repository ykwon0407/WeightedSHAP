'''
Reference: https://github.com/iancovert/removal-explanations/blob/main/rexplain/behavior.py
'''
import numpy as np

class PredictionGame:
    '''
    Cooperative game for an individual example's prediction.

    Args:
      extension: model extension (see removal.py).
      sample: numpy array representing a single model input.
    '''

    def __init__(self, extension, sample, superpixel_size=1):
        # Add batch dimension to sample.
        if sample.ndim == 1:
            sample = sample[np.newaxis]
        # elif sample.shape[0] != 1:
        # raise ValueError('sample must have shape (ndim,) or (1,ndim)')

        self.extension = extension
        self.sample = sample
        self.players = np.prod(sample.shape)//(superpixel_size**2)//sample.shape[0] # sample.shape[1]

        # Caching.
        self.sample_repeat = sample

    def __call__(self, S):
        # Return scalar if single subset.
        single_eval = (S.ndim == 1)
        if single_eval:
            S = S[np.newaxis]
            input_data = self.sample
        else:
            # Try to use caching for repeated data.
            if len(S) != len(self.sample_repeat):
                self.sample_repeat = self.sample.repeat(len(S), 0)
            input_data = self.sample_repeat

        # Evaluate.
        output = self.extension(input_data, S)
        if single_eval:
            output = output[0]
        return output

def crossentropyloss(pred, target):
    '''Cross entropy loss that does not average across samples.'''
    if pred.ndim == 1:
        pred = pred[:, np.newaxis]
        pred = np.concatenate((1 - pred, pred), axis=1)

    if pred.shape == target.shape:
        # Soft cross entropy loss.
        pred = np.clip(pred, a_min=1e-12, a_max=1-1e-12)
        return - np.sum(np.log(pred) * target, axis=1)
    else:
        # Standard cross entropy loss.
        return - np.log(pred[np.arange(len(pred)), target])


def mseloss(pred, target):
    '''MSE loss that does not average across samples.'''
    return np.sum((pred - target) ** 2, axis=1)

class PredictionLossGame:
    '''
    Cooperative game for an individual example's loss value.
    Args:
      extension: model extension (see removal.py).
      sample: numpy array representing a single model input.
      label: the input's true label.
      loss: loss function (see utils.py).
    '''
    def __init__(self, extension, sample, label, loss=mseloss):
        # Add batch dimension to sample.
        if sample.ndim == 1:
            sample = sample[np.newaxis]

        # Add batch dimension to label.
        if np.isscalar(label):
            label = np.array([label])

        # Convert label dtype if necessary.
        if loss is crossentropyloss:
            # Make sure not soft cross entropy.
            if (label.ndim <= 1) or (label.shape[1] == 1):
                # Only convert if float.
                if np.issubdtype(label.dtype, np.floating):
                    label = label.astype(int)

        self.extension = extension
        self.sample = sample
        self.label = label
        self.loss = loss
        self.players = sample.shape[1]

        # Caching.
        self.sample_repeat = sample
        self.label_repeat = label

    def __call__(self, S):
        # Return scalar if single subset.
        single_eval = (S.ndim == 1)
        if single_eval:
            S = S[np.newaxis]
            input_data = self.sample
            output_label = self.label
        else:
            # Try to use caching for repeated data.
            if len(S) != len(self.sample_repeat):
                self.sample_repeat = self.sample.repeat(len(S), 0)
                self.label_repeat = self.label.repeat(len(S), 0)
            input_data = self.sample_repeat
            output_label = self.label_repeat

        # Evaluate.
        output = - self.loss(self.extension(input_data, S), output_label)
        if single_eval:
            output = output[0]
        return output


class DatasetLossGame:
    '''
    Cooperative game representing the model's loss over a dataset.
    TODO: this implementation is slower than SAGE because it averages
    loss over entire dataset for each S. Need to reimplement as a stochastic
    game (with caching) to accelerate convergence.
    Args:
      extension: model extension (see removal.py).
      data: array of model inputs.
      labels: array of corresponding labels.
      loss: loss function (see utils.py).
    '''
    def __init__(self, extension, data, labels, loss):
        # Convert labels dtype if necessary.
        if loss is crossentropyloss:
            # Make sure not soft cross entropy.
            if (labels.ndim == 1) or (labels.shape[1] == 1):
                # Only convert if float.
                if np.issubdtype(labels.dtype, np.floating):
                    labels = labels.astype(int)

        self.extension = extension
        self.data = data
        self.labels = labels
        self.loss = loss
        self.players = data.shape[1]
        self.data_tile = self.data
        self.label_tile = self.labels

    def __call__(self, S):
        # Return scalar is single subset.
        single_eval = (S.ndim == 1)
        if single_eval:
            S = S[np.newaxis]

        # Prepare data.
        if len(self.data_tile) != len(self.data) * len(S):
            self.data_tile = np.tile(self.data, (len(S), 1))
            self.label_tile = np.tile(
                self.labels,
                (len(S), *[1 for _ in range(len(self.labels.shape[1:]))]))
        S = S.repeat(len(self.data), 0)

        # Evaluate.
        output = - self.loss(self.extension(self.data_tile, S), self.label_tile)
        output = output.reshape((-1, self.data.shape[0]))
        output = np.mean(output, axis=1)
        if single_eval:
            output = output[0]
        return output

        