import torch
import numpy as np
import numpy.random as npr

from torchvision.transforms import Compose
from utils import Rescale, Normailize, Reshape

# TODO: keep_ratio

class Collater(object):
    """"""
    def __init__(self, scales, keep_ratio=False, multiple=32):
        if isinstance(scales, (int, float)):
            self.scales = np.array([scales], dtype=np.int32)
        else:
            self.scales = np.array(scales, dtype=np.int32)
        self.keep_ratio = keep_ratio
        self.multiple = multiple

    def __call__(self, batch):
        random_scale_inds = npr.randint(0, high=len(self.scales))
        target_size = self.scales[random_scale_inds]
        target_size = int(np.floor(float(target_size) / self.multiple) * self.multiple)
        rescale = Rescale(target_size=target_size, keep_ratio=self.keep_ratio)
        transform = Compose([Normailize(), Reshape(unsqueeze=False)])

        images = [sample['image'] for sample in batch]
        existence = [sample['existence'] for sample in batch]
        path = [sample['path'] for sample in batch]
        batch_size = len(images)
        max_width, max_height = -1, -1
        for i in range(batch_size):
            im, _ = rescale(images[i])
            height, width = im.shape[0], im.shape[1]
            max_width = width if width > max_width else max_width
            max_height = height if height > max_height else max_height

        padded_ims = torch.zeros(batch_size, 3, max_height, max_width)

        for i in range(batch_size):
            im = images[i]
            im, im_scale = rescale(im)
            height, width = im.shape[0], im.shape[1]
            padded_ims[i, :, :height, :width] = transform(im)
        return {'image': padded_ims, 'existence': existence, 'path': path}
