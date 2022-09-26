'''
Reference: https://github.com/iancovert/fastshap/blob/main/fastshap/image_imputers.py
'''
import torch.nn as nn

class ImageImputer:
    '''
    Image imputer base class.
    Args:
      width: image width.
      height: image height.
      superpixel_size: superpixel width/height (int).
    '''

    def __init__(self, width, height, superpixel_size=1):
        # Verify arguments.
        assert width % superpixel_size == 0
        assert height % superpixel_size == 0

        # Set up superpixel upsampling.
        self.width = width
        self.height = height
        self.supsize = superpixel_size
        if superpixel_size == 1:
            self.upsample = nn.Identity()
        else:
            self.upsample = nn.Upsample(
                scale_factor=superpixel_size, mode='nearest')

        # Set up number of players.
        self.small_width = width // superpixel_size
        self.small_height = height // superpixel_size
        self.num_players = self.small_width * self.small_height

    def __call__(self, x, S):
        '''
        Evaluate with subset of features.
        Args:
          x: input examples.
          S: coalitions.
        '''
        raise NotImplementedError

    def resize(self, S):
        '''
        Resize coalition variable S into grid shape.
        Args:
          S: coalitions.
        '''
        if len(S.shape) == 2:
            S = S.reshape(S.shape[0], self.small_height,
                          self.small_width).unsqueeze(1)
        return self.upsample(S)


class BaselineImageImputer(ImageImputer):
    '''
    Evaluate image model while replacing features with baseline values.
    Args:
      model: predictive model.
      baseline: baseline value(s).
      width: image width.
      height: image height.
      superpixel_size: superpixel width/height (int).
      link: link function (e.g., nn.Softmax).
    '''

    def __init__(self, model, baseline, width, height, superpixel_size,
                 link=None):
        super().__init__(width, height, superpixel_size)
        self.model = model
        self.baseline = baseline

        # Set up link.
        if link is None:
            self.link = nn.Identity()
        elif isinstance(link, nn.Module):
            self.link = link
        else:
            raise ValueError('unsupported link function: {}'.format(link))

    def __call__(self, x, S):
        '''
        Evaluate model using baseline values.
        '''
        S = self.resize(S)
        x_baseline = S * x + (1 - S) * self.baseline
        return self.link(self.model(x_baseline))

        