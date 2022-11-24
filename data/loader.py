from pathlib import Path
import numpy as np
from fume import calculate_fundamental_matrix
from torch.utils.data import Dataset
import torch
from torchvision.transforms import Normalize
from scipy.signal import convolve2d

# A. S. Wang et al., “Low-dose preview for patient-specific, task-specific technique selection in cone-beam CT,
# ” Med Phys, vol. 41, no. 7, Jul. 2014.
kernel_shot_noise = np.array([[0.03, 0.06, 0.02],
                              [0.11, 0.98, 0.11],
                              [0.02, 0.06, 0.03]])


def add_noise(proj_stack, photon_num=None):
    """photon number usually in [10e1, 10e3]"""
    output = proj_stack.copy()
    photon_number = np.random.randint(10e1, 10e3) if photon_num is None else photon_num  # the higher the less noisy

    # apply shot noise as in deepdrr
    for i, p in enumerate(proj_stack):
        lamb = p * photon_number  # lambda parameter for poisson dist
        shot_noise = ((np.random.poisson(lamb) - lamb) / lamb) * p
        output[i] += convolve2d(shot_noise, kernel_shot_noise, mode="same")

    return output

def downsample_tensor(ten: torch.Tensor, d, p):
    return torch.nn.functional.pad(ten[:, ::d, ::d], (p, p, p, p), mode="constant")


def neglog(intensities):
    return -np.log(np.maximum(intensities, np.finfo(dtype=intensities.dtype).eps))


class NPZData(Dataset):
    def __init__(self, path, downsample=1, pad=0, binarize=False, noise=False, eval=False):
        self.path = path
        self.files = sorted(list(Path(path).glob('*.npz')))
        self.downsample, self.pad, self.binarize, self.noise, self.eval = downsample, pad, binarize, noise, eval

        # calculate how to adapt the projection matrices
        centered = np.eye(3)
        centered[:2, 2] = - (976 / 2 - 0.5)  # the Siemens matrices map onto detector of size 976
        scale = np.diag([1 / self.downsample, 1 / self.downsample, 1])  # scales the detector size
        FLIPLR = np.asarray([[-1, 0, 975],
                             [0, 1, 0],
                             [0, 0, 1]])  # specifically for Siemens matrices
        self.adapt = scale @ centered @ FLIPLR

    def __len__(self):
        return len(self.files)

    def __getitem__(self, item):
        # x are normalized intensities in range (0, 1)
        sample = np.load(str(self.files[item]))
        x, y = sample['x'], sample['y']

        # adding noise to input images
        if self.noise:
            x = add_noise(sample['x'])

        if self.binarize:
            y[y > 0] = 1

        # images need to be transposed for fume layers
        x_tensor = torch.from_numpy(np.transpose(x, axes=(0, 2, 1)))
        y_tensor = torch.from_numpy(np.transpose(y, axes=(0, 2, 1)))

        # normalize
        x_tensor = Normalize(torch.mean(x_tensor, dim=(1, 2)), torch.std(x_tensor, dim=(1, 2)), inplace=True)(x_tensor)

        # downsample and augment etc.
        if self.downsample != 1 or self.pad != 0:
            x_tensor = downsample_tensor(x_tensor, self.downsample, self.pad)
            y_tensor = downsample_tensor(y_tensor, self.downsample, self.pad)

        # adapt projection matrices to shape
        P1 = self.adapt @ sample['P'][0]
        P2 = self.adapt @ sample['P'][1]

        # compute fundamental matrices from projection matrices
        F12 = calculate_fundamental_matrix(P_src=P1, P_dst=P2)
        F21 = calculate_fundamental_matrix(P_src=P2, P_dst=P1)
        P_tensor = torch.tensor(np.array([F12, F21]), dtype=torch.float32)

        if not self.eval:
            return x_tensor, P_tensor, y_tensor
        else:
            return x_tensor, P_tensor, y_tensor, str(self.files[item])
