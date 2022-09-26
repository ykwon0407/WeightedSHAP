'''
Reference: https://github.com/iancovert/removal-explanations/blob/main/rexplain/removal.py
'''
import numpy as np

class MarginalExtension:
    '''Extend a model by marginalizing out removed features using their
    marginal distribution.'''
    def __init__(self, data, model):
        self.model = model
        self.data = data
        self.data_repeat = data
        self.samples = len(data)
        # self.x_addr = None
        # self.x_repeat = None

    def __call__(self, x, S):
        # Prepare x and S.
        n = len(x)
        x = x.repeat(self.samples, 0)
        S = S.repeat(self.samples, 0)

        # Prepare samples.
        if len(self.data_repeat) != self.samples * n:
            self.data_repeat = np.tile(self.data, (n, 1))

        # Replace specified indices.
        x_ = x.copy()
        x_[~S] = self.data_repeat[~S]

        # Make predictions.
        pred = self.model(x_)
        pred = pred.reshape(-1, self.samples, *pred.shape[1:])
        return np.mean(pred, axis=1)


class ConditionalSupervisedExtension:
    '''Extend a model using a supervised surrogate model.'''
    def __init__(self, surrogate):
        self.surrogate = surrogate

    def __call__(self, x, S):
        return self.surrogate(x, S)

class DefaultExtension:
    '''Extend a model by replacing removed features with default values.'''
    def __init__(self, values, model):
        self.model = model
        if values.ndim == 1:
            values = values[np.newaxis]
        elif values[0] != 1:
            raise ValueError('values shape must be (dim,) or (1, dim)')
        self.values = values
        self.values_repeat = values

    def __call__(self, x, S):
        # Prepare x.
        if len(x) != len(self.values_repeat):
            self.values_repeat = self.values.repeat(len(x), 0)

        # Replace specified indices.
        x_ = x.copy()
        x_[~S] = self.values_repeat[~S]

        # Make predictions.
        return self.model(x_)

class MarginalExtensionApprox:
    '''Extend a model by marginalizing out removed features using their
    marginal distribution.'''
    def __init__(self, data_mean, model, grad_array):
        self.model = model
        self.data_mean = data_mean
        self.grad=grad_array

    def __call__(self, x, S):
        # Prepare samples.
        n=len(x)
        if len(self.data_mean) != n:
            self.data_repeat = np.tile(self.data_mean, (n, 1))

        # Replace specified indices.
        x_ = x.copy()
        x_[~S] = self.data_repeat[~S]

        # Make predictions.
        pred = self.model(x)
        pred += (x_-x).dot(self.grad)
        pred = pred.reshape(-1, 1, *pred.shape[1:])
        return np.mean(pred, axis=1)

class UniformExtension:
    '''Extend a model by marginalizing out removed features using a
    uniform distribution.'''
    def __init__(self, values, categorical_inds, samples, model):
        self.model = model
        self.values = values
        self.categorical_inds = categorical_inds
        self.samples = samples

    def __call__(self, x, S):
        # Prepare x and S.
        n = len(x)
        x = x.repeat(self.samples, 0)
        S = S.repeat(self.samples, 0)

        # Prepare samples.
        samples = np.zeros((n * self.samples, x.shape[1]))
        for i in range(x.shape[1]):
            if i in self.categorical_inds:
                inds = np.random.choice(
                    len(self.values[i]), n * self.samples)
                samples[:, i] = self.values[i][inds]
            else:
                samples[:, i] = np.random.uniform(
                    low=self.values[i][0], high=self.values[i][1],
                    size=n * self.samples)

        # Replace specified indices.
        x_ = x.copy()
        x_[~S] = samples[~S]

        # Make predictions.
        pred = self.model(x_)
        pred = pred.reshape(-1, self.samples, *pred.shape[1:])
        return np.mean(pred, axis=1)


class UniformContinuousExtension:
    '''
    Extend a model by marginalizing out removed features using a
    uniform distribution. Specific to sets of continuous features.

    TODO: should we have caching here for repeating x?

    '''
    def __init__(self, min_vals, max_vals, samples, model):
        self.model = model
        self.min = min_vals
        self.max = max_vals
        self.samples = samples

    def __call__(self, x, S):
        # Prepare x and S.
        x = x.repeat(self.samples, 0)
        S = S.repeat(self.samples, 0)

        # Prepare samples.
        u = np.random.uniform(size=x.shape)
        samples = u * self.min + (1 - u) * self.max

        # Replace specified indices.
        x_ = x.copy()
        x_[~S] = samples[~S]

        # Make predictions.
        pred = self.model(x_)
        pred = pred.reshape(-1, self.samples, *pred.shape[1:])
        return np.mean(pred, axis=1)


class ProductMarginalExtension:
    '''Extend a model by marginalizing out removed features the
    product of their marginal distributions.'''
    def __init__(self, data, samples, model):
        self.model = model
        self.data = data
        self.data_repeat = data
        self.samples = samples

    def __call__(self, x, S):
        # Prepare x and S.
        n = len(x)
        x = x.repeat(self.samples, 0)
        S = S.repeat(self.samples, 0)

        # Prepare samples.
        samples = np.zeros((n * self.samples, x.shape[1]))
        for i in range(x.shape[1]):
            inds = np.random.choice(len(self.data), n * self.samples)
            samples[:, i] = self.data[inds, i]

        # Replace specified indices.
        x_ = x.copy()
        x_[~S] = samples[~S]

        # Make predictions.
        pred = self.model(x_)
        pred = pred.reshape(-1, self.samples, *pred.shape[1:])
        return np.mean(pred, axis=1)


class SeparateModelExtension:
    '''Extend a model using separate models for each subset of features.'''
    def __init__(self, model_dict):
        self.model_dict = model_dict

    def __call__(self, x, S):
        output = []
        for i in range(len(S)):
            # Extract model.
            row = S[i]
            model = self.model_dict[str(row)]

            # Make prediction.
            output.append(model(x[i:i+1, row]))

        return np.concatenate(output, axis=0)


class ConditionalExtension:
    '''Extend a model by marginalizing out removed features using a model of
    their conditional distribution.'''
    def __init__(self, conditional_model, samples, model):
        self.model = model
        self.conditional_model = conditional_model
        self.samples = samples
        self.x_addr = None
        self.x_repeat = None

    def __call__(self, x, S):
        # Prepare x.
        if self.x_addr != id(x):
            self.x_addr = id(x)
            self.x_repeat = x.repeat(self.samples, 0)
        x = self.x_repeat

        # Prepare samples.
        S = S.repeat(self.samples, 0)
        samples = self.conditional_model(x, S)

        # Replace specified indices.
        x_ = x.copy()
        x_[~S] = samples[~S]

        # Make predictions.
        pred = self.model(x_)
        pred = pred.reshape(-1, self.samples, *pred.shape[1:])
        return np.mean(pred, axis=1)



