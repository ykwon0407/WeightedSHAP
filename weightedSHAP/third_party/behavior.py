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

