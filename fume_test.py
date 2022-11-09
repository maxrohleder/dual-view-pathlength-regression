import torch
from fume import Fume3dLayer
from fume import calculate_fundamental_matrix
import numpy as np
from matplotlib import pyplot as plt
from tifffile import imsave

scale = np.diag([2, 2, 1])
TRANSPOSEUV = np.asarray([[0, 1, 0],
                          [1, 0, 0],
                          [0, 0, 1]])
FLIPLR = np.asarray([[-1, 0, 975],
                     [0, 1, 0],
                     [0, 0, 1]])
centered = np.eye(3)
centered[:2, 2] = - np.asarray([487.5, 487.5])

sample = np.load(r"/media/dl/data2/pathlength-reg/train/Spine03_0_180_id220.npz")
y = sample['y']
view1 = torch.from_numpy(y[0].T.reshape((1, 1, 976, 976)).copy()).cuda()
view2 = torch.from_numpy(y[1].T.reshape((1, 1, 976, 976)).copy()).cuda()

# P1 = centered @ sample['P'][0]
# P2 = centered @ sample['P'][1]
P1 = centered @ FLIPLR @ sample['P'][0]
P2 = centered @ FLIPLR @ sample['P'][1]

# compute fundamental matrices from projection matrices
F12 = torch.from_numpy(calculate_fundamental_matrix(P_src=P1, P_dst=P2).reshape((1, 3, 3)).astype(np.float32)).cuda()
F21 = torch.from_numpy(calculate_fundamental_matrix(P_src=P2, P_dst=P1).reshape((1, 3, 3)).astype(np.float32)).cuda()

f = Fume3dLayer()
CM1 = f(view2, F12, F21)
CM2 = f(view1, F21, F12)

plt.figure(figsize=(8, 4))
plt.subplot(121)
plt.imshow(view1.cpu().numpy().squeeze().T, cmap="Greys")
plt.imshow(CM1.cpu().numpy().squeeze().T, alpha=0.5, cmap="Greens")

plt.subplot(122)
plt.imshow(view2.cpu().numpy().squeeze().T, cmap="Greys")
plt.imshow(CM2.cpu().numpy().squeeze().T, alpha=0.5, cmap="Greens")
plt.tight_layout()

plt.show()


imsave("view1.tif", view1.cpu().numpy().squeeze().T)
imsave("CM1.tif", CM1.cpu().numpy().squeeze().T)
imsave("view2.tif", view2.cpu().numpy().squeeze().T)
imsave("CM2.tif", CM2.cpu().numpy().squeeze().T)
