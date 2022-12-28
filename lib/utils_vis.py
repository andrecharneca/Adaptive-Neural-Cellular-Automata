import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class SamplePool:
    def __init__(self, *, _parent=None, _parent_idx=None, **slots):
        self._parent = _parent
        self._parent_idx = _parent_idx
        self._slot_names = slots.keys()
        self._size = None
        for k, v in slots.items():
            if self._size is None:
                self._size = len(v)
            assert self._size == len(v)
            setattr(self, k, np.asarray(v))

    def sample(self, n):
        idx = np.random.choice(self._size, n, False)
        batch = {k: getattr(self, k)[idx] for k in self._slot_names}
        batch = SamplePool(**batch, _parent=self, _parent_idx=idx)
        return batch

    def commit(self):
        for k in self._slot_names:
            getattr(self._parent, k)[self._parent_idx] = getattr(self, k)

def to_alpha(x):
    return np.clip(x[..., 3:4], 0, 0.9999)

def to_rgb(x):
    # assume rgb premultiplied by alpha
    rgb, a = x[..., :3], to_alpha(x)
    return np.clip(1.0-a+rgb, 0, 0.9999)

def get_living_mask(x):
    return nn.MaxPool2d(3, stride=1, padding=1)(x[:, 3:4, :, :])>0.1

def make_seeds(shape, n_channels, n=1):
    x = np.zeros([n, shape[0], shape[1], n_channels], np.float32)
    x[:, shape[0]//2, shape[1]//2, 3:] = 1.0
    return x

def make_seed(shape, n_channels):
    seed = np.zeros([shape[0], shape[1], n_channels], np.float32)
    seed[shape[0]//2, shape[1]//2, 3:] = 1.0
    return seed

def make_circle_masks(n, h, w, location='random'):
    """
    Make n circular masks at location, to be applied to image of size (h,w)
    """
    assert location in ['random', 'center', 'head', 'leg1', 'tail'] 

    x = np.linspace(-1.0, 1.0, w)[None, None, :]
    y = np.linspace(-1.0, 1.0, h)[None, :, None]
    center = np.ones([2,n,1,1])
    if location=='random':
        center = np.random.random([2,n,1,1])*1.0-0.5
        r = np.random.random([n, 1, 1])*0.3+0.1
    elif location=='head':
        center[0] = -0.45; center[1] = -0.3
        r = np.ones([n, 1, 1])*0.25
    elif location=='leg1':
        center[0] = -0.08; center[1] = -0.35
        r = np.ones([n, 1, 1])*0.05 + 0.1  
    elif location=='tail':
        center[0] = 0.4; center[1] = 0.3
        r = np.ones([n, 1, 1])*0.25
    elif location=='center':
        center[0] = -0.1; center[1] = 0.
        r = np.ones([n, 1, 1])*0.12
    x, y = (x-center[0])/r, (y-center[1])/r
    mask = (x*x+y*y < 1.0).astype(np.float32)
    return mask

def damage_batch(x_batch, device, img_size = 72, damage_location = 'random', damaged_in_batch = 1):
  """
  Apply circular damage to some images in the batch, at the given location.

  x_batch.shape = (batch_size, img_size, img_size, channels_n)
  """
  x_batch_copy = x_batch.clone()
  damage = 1.0-make_circle_masks(damaged_in_batch, img_size, img_size, damage_location)[..., None]
  damage = torch.from_numpy(damage).to(device)
  x_batch_copy[:damaged_in_batch]*=damage

  return x_batch_copy
