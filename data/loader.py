from pathlib import Path
import numpy as np
from fume import calculate_fundamental_matrix
from torch.utils.data import Dataset
import torch


FLIPLR = np.asarray([[-1, 0, 975],
                     [0, 1, 0],
                     [0, 0, 1]])


def downsample_tensor(ten: torch.Tensor, d, p):
    return torch.nn.functional.pad(ten[:, ::d, ::d], (p, p, p, p), mode="constant")

class NPZData(Dataset):
    def __init__(self, path, downsample=1, pad=0):
        self.path = path
        self.files = sorted(list(Path(path).glob('*.npz')))
        self.downsample, self.pad = downsample, pad

    def __len__(self):
        return len(self.files)

    def __getitem__(self, item):
        sample = np.load(str(self.files[item]))
        x_shape = sample['x'].shape

        # images need to be transposed for fume layers
        x_tensor = torch.from_numpy(np.transpose(sample['x'], axes=(0, 2, 1)))
        y_tensor = torch.from_numpy(np.transpose(sample['y'], axes=(0, 2, 1)))

        # downsample and augment etc.
        if self.downsample != 1 or self.pad != 0:
            x_tensor = downsample_tensor(x_tensor, self.downsample, self.pad)
            y_tensor = downsample_tensor(y_tensor, self.downsample, self.pad)

        # adapt projection matrices to shape
        centered = np.eye(3)
        centered[:2, 2] = - ((np.asarray(x_shape[1:]) / 2) - 0.5)
        scale = np.diag([1 / self.downsample, 1 / self.downsample, 1])
        P1 = scale @ centered @ FLIPLR @ sample['P'][0]
        P2 = scale @ centered @ FLIPLR @ sample['P'][1]

        # compute fundamental matrices from projection matrices
        F12 = calculate_fundamental_matrix(P_src=P1, P_dst=P2)
        F21 = calculate_fundamental_matrix(P_src=P2, P_dst=P1)
        P_tensor = torch.tensor(np.array([F12, F21]), dtype=torch.float32)

        return x_tensor, P_tensor, y_tensor
