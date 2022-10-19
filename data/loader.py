from pathlib import Path
import numpy as np
from fume import calculate_fundamental_matrix
from torch.utils.data import Dataset
import torch


class NPZData(Dataset):
    def __init__(self, path, x_transform=None, y_transform=None):
        self.path = path
        self.files = sorted(list(Path(path).glob('*.npz')))
        self.x_transform, self.y_transform = x_transform, y_transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, item):
        sample = np.load(str(self.files[item]))

        x_tensor = torch.from_numpy(sample['x'])
        y_tensor = torch.from_numpy(sample['y'])

        # compute fundamental matrices from projection matrices
        F12 = calculate_fundamental_matrix(P_src=sample['P'][0], P_dst=sample['P'][1])
        F21 = calculate_fundamental_matrix(P_src=sample['P'][1], P_dst=sample['P'][0])
        P_tensor = torch.tensor(np.array([F12, F21]), dtype=torch.float32)

        if self.x_transform is not None:
            x_tensor = self.x_transform(x_tensor)
        if self.y_transform is not None:
            y_tensor = self.y_transform(y_tensor)

        return x_tensor, P_tensor, y_tensor
