"""
Code inherited from old BraTS code. Needs updating/ rewriting.
"""

import collections
import numpy as np

class Base(object):
    def sample(self, *shape):
        return shape

    def tf(self, img, k=0):
        return img

    def __call__(self, img, dim=3, reuse=False): # class -> func()
        # image: nhwtc
        # shape: no first dim
        if not reuse:
            im = img if isinstance(img, np.ndarray) else img[0]
            # how to know  if the last dim is channel??
            # nhwtc vs nhwt??
            shape = im.shape[1:dim+1]
            # print(dim,shape) # 3, (240,240,155)
            self.sample(*shape)

        if isinstance(img, collections.Sequence):
            return [self.tf(x, k) for k, x in enumerate(img)] # img:k=0,label:k=1

        return self.tf(img)

    def __str__(self):
        return 'Identity()'

class CenterCrop(Base):
    def __init__(self, size):
        self.size = size
        self.buffer = None

    def sample(self, *shape):
        size = self.size
        start = [(s -size)//2 for s in shape]
        self.buffer = [slice(None)] + [slice(s, s+size) for s in start]
        return [size] * len(shape)

    def tf(self, img, k=0):
        # print(img.shape)#(1, 240, 240, 155, 4)
        return img[tuple(self.buffer)]
        # return img[self.buffer]

    def __str__(self):
        return 'CenterCrop({})'.format(self.size)

class CenterCropBySize(CenterCrop):
    def sample(self, *shape):
        assert len(self.size) == 3  # random crop [H,W,T] from img [240,240,155]
        if not isinstance(self.size, list):
            size = list(self.size)
        else:
            size = self.size
        start = [(s-i)//2 for i, s in zip(size, shape)]
        self.buffer = [slice(None)] + [slice(s, s+i) for i, s in zip(size, start)]
        return size

    def __str__(self):
        return 'CenterCropBySize({})'.format(self.size)

class NumpyType(Base):
    def __init__(self, types, num=-1):
        self.types = types # ('float32', 'int64')
        self.num = num

    def tf(self, img, k=0):
        if self.num > 0 and k >= self.num:
            return img
        # make this work with both Tensor and Numpy
        return img.astype(self.types[k])

    def __str__(self):
        s = ', '.join([str(s) for s in self.types])
        return 'NumpyType(({}))'.format(s)