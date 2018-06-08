from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import inspect, re
import numpy as np
import os
import collections

colormap = np.array([[0, 0, 0], 
                     [0.5, 0, 0],
                     [0.9961, 0, 0],
                     [0, 0.332, 0],
                     [0.6641, 0, 0.1992],
                     [0.9961, 0.332, 0],
                     [0, 0, 0.332],
                     [0, 0.4648, 0.8633],
                     [0.332, 0.332, 0],
                     [0, 0.332, 0.332],
                     [0.332, 0.1992, 0],
                     [0.2031, 0.3359, 0.5],
                     [0, 0.5, 0],
                     [0, 0, 0.9961],
                     [0.1992, 0.6641, 0.8633],
                     [0, 0.9961, 0.9961],
                     [0.332, 0.9961, 0.6641],
                     [0.6641, 0.9961, 0.332],
                     [0.9961, 0.9961, 0],
                     [0.9961, 0.6641, 0]])

# Converts a Segmentation Map into rgb image of numpy array
def segmap2im(map_tensor):
    map_numpy = map_tensor[0].cpu().float().numpy()
    c, h, w = map_numpy.shape
    rgbimg = np.zeros((h, w, 3), dtype=np.uint8)
    for i in range(c):
        rgbimg[map_numpy[i,:,:]==1.0,:] += (colormap[i, :] * 255.0).astype(np.uint8)
    return rgbimg

# Converts a Segmentation Map into label mask of tensor
def segmap2mask(map_tensor):
    map_numpy = map_tensor.cpu().float().numpy()
    b, c, h, w = map_numpy.shape
    labelmask = np.zeros((b, 1, h, w), dtype=np.uint8)
    for i in range(b):
        for j in range(c):
            labelmask[i, 0, map_numpy[i,j,:,:]==1.0] += j
    labelmask = torch.from_numpy(labelmask)
    return labelmask


# Converts a Mask into rgb image of numpy array
def mask2im(mask_tensor):
    mask_numpy = mask_tensor[0].cpu().float().numpy()
    c, h, w = mask_numpy.shape
    rgbimg = np.zeros((h, w, 3), dtype=np.uint8)
    for i in range(20):
        rgbimg[mask_numpy[0,:,:]==i,:] += (colormap[i, :] * 255.0).astype(np.uint8)
    return rgbimg


# Converts a Tensor into a Numpy array
# |imtype|: the desired type of the converted numpy array
def tensor2im(image_tensor, imtype=np.uint8):
    image_numpy = image_tensor[0].cpu().float().numpy()

    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))

    image_numpy = (image_numpy + 1) / 2.0 * 255.0

    return np.transpose(image_numpy,(1,2,0)).astype(imtype)

def keyp2im(keyp_tensor, imtype=np.uint8):
    keyp_numpy = keyp_tensor[0].cpu().float().numpy()
    keyp_numpy = (keyp_numpy + 1) / 2.0
    keyp_numpy = np.clip(np.sum(keyp_numpy[0:14,:,:], axis=0), 0, 1)
    keyp_numpy = np.tile(keyp_numpy, (3, 1, 1))
    keyp_numpy = np.transpose(keyp_numpy, (1, 2, 0)) * 255.0

    return keyp_numpy.astype(imtype)

def diagnose_network(net, name='network'):
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)

def info(object, spacing=10, collapse=1):
    """Print methods and doc strings.
    Takes module, class, list, dictionary, or string."""
    methodList = [e for e in dir(object) if isinstance(getattr(object, e), collections.Callable)]
    processFunc = collapse and (lambda s: " ".join(s.split())) or (lambda s: s)
    print( "\n".join(["%s %s" %
                     (method.ljust(spacing),
                      processFunc(str(getattr(object, method).__doc__)))
                     for method in methodList]) )

def varname(p):
    for line in inspect.getframeinfo(inspect.currentframe().f_back)[3]:
        m = re.search(r'\bvarname\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)', line)
        if m:
            return m.group(1)

def print_numpy(x, val=True, shp=False):
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
