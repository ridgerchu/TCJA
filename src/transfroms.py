import random
import numpy as np
import torch
from torchvision.transforms import functional as F
import torchvision
import math
#torchvision.transforms.RandomAffine(degrees, translate=None, scale=None, shear=None, resample=False, fillcolor=0)
def dvs_aug(data):

    flip = random.random() > 0.5
    if flip:
        data = np.flip(data, axis=3)
    off1 = random.randint(-5, 5)
    off2 = random.randint(-5, 5)
    data = np.roll(data, shift=(off1, off2), axis=(2, 3))
    return data

class RandomCompose:
    """Composes several transforms together. This transform does not support torchscript.
    Please, see the note below.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.

    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.PILToTensor(),
        >>>     transforms.ConvertImageDtype(torch.float),
        >>> ])

    .. note::
        In order to script the transformations, please use ``torch.nn.Sequential`` as below.

        >>> transforms = torch.nn.Sequential(
        >>>     transforms.CenterCrop(10),
        >>>     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        >>> )
        >>> scripted_transforms = torch.jit.script(transforms)

        Make sure to use only scriptable transformations, i.e. that work with ``torch.Tensor``, does not require
        `lambda` functions or ``PIL.Image``.

    """

    def __init__(self, const_transforms, random_transforms, select_num):
        self.random_transforms = random_transforms
        self.const_transfroms = const_transforms
        self.select_num = select_num

    def __call__(self, img):
        sample = random.sample(self.random_transforms, self.select_num)
        for t in self.const_transfroms:
            img = t(img)
        for t in sample:
            img = t(img)
        img = np.ascontiguousarray(img)
        return img

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string

class Flip(object):
    """增加椒盐噪声
    Args:
        snr （float）: Signal Noise Rate
        p (float): 概率值，依概率执行该操作
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, data):
        flip = random.random() > self.p
        if flip:
            data = np.flip(data, axis=3)
        return data

class Resize(object):
    """增加椒盐噪声
    Args:
        snr （float）: Signal Noise Rate
        p (float): 概率值，依概率执行该操作
    """

    def __init__(self, resolution):
        self.resize = torchvision.transforms.Resize(size=resolution)  # 48 48
        self.imgx = torchvision.transforms.ToPILImage()

    def __call__(self, data):
        new_data = []
        data = torch.tensor(np.ascontiguousarray(data))
        for t in range(data.shape[0]):
            new_data.append((np.asarray(self.resize(self.imgx(data[t, ...])))).transpose())
        data = np.stack(new_data, axis=0)
        return data.astype('float64')


class Rolling(object):
    """增加椒盐噪声
    Args:
        snr （float）: Signal Noise Rate
        p (float): 概率值，依概率执行该操作
    """

    def __init__(self, bias_pos=5, bias_neg=-5):
        self.bias_pos = bias_pos
        self.bias_neg = bias_neg

    def __call__(self, data):
        off1 = random.randint(self.bias_neg, self.bias_pos)
        off2 = random.randint(self.bias_neg, self.bias_pos)
        data = np.roll(data, shift=(off1, off2), axis=(2, 3))
        return data
    
class Rotation(object):
    """增加椒盐噪声
    Args:
        snr （float）: Signal Noise Rate
        p (float): 概率值，依概率执行该操作
    """

    def __init__(self, angle=15):
        self.angle = angle


    def __call__(self, data):
        # Define the most occuring variables
        return_data = []
        '''angle = random.randint(-self.angle, self.angle)
        data = torch.Tensor(data)
        new_data = []
        for i in range(data.shape[0]):

            temp_matrix = data[i, :, :, :]
            temp_matrix = F.rotate(temp_matrix,angle,fill=0)
            new_data.append(temp_matrix)
        new_data = torch.stack(new_data,dim=0)
        return new_data.numpy()'''
        angle = random.randint(-self.angle, self.angle)
        data = np.ascontiguousarray(data)
        data = torch.Tensor(data)
        data = F.rotate(data, angle=angle, fill=0)
        return data.numpy()

class Cutout(object):
    """增加椒盐噪声
    Args:
        snr （float）: Signal Noise Rate
        p (float): 概率值，依概率执行该操作
    """

    def __init__(self,max_length = 8):
        self.max_length = max_length


    def __call__(self, data):
        # Define the most occuring variables
        length = random.randint(1, self.max_length)
        event_height = data.shape[2]
        ceil_int = math.ceil(length/2)
        center = (random.randint(ceil_int, event_height-ceil_int), random.randint(ceil_int, event_height-ceil_int))
        data[:, :, center[0]:center[0] + length, center[1]:center[1] + length] = 0
        return data

class XShear(object):
    """增加椒盐噪声
    Args:
        snr （float）: Signal Noise Rate
        p (float): 概率值，依概率执行该操作
    """

    def __init__(self, angle=8):
        self.angle = angle


    def __call__(self, data):
        # Define the most occuring variables
        angle = random.randint(-self.angle, self.angle)
        data = np.ascontiguousarray(data)
        data = torch.Tensor(data)
        data = F.affine(data,angle=0,scale=1,shear=[angle,0],fill=0,translate=(0,0))
        return data.numpy()

def mixup_data(x, y, alpha=0.5, use_cuda=True):

    '''Compute the mixup data. Return mixed inputs, pairs of targets, and lambda'''
    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.
    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index,:]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(y_a, y_b, lam):
    return lambda criterion, pred: lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

