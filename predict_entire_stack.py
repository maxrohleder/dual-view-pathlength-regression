from pathlib import Path

import matplotlib.pyplot as plt
from tifffile import imread, imsave
import torch
from torch.utils.data import DataLoader

from data.loader import NPZData
from data.utils import spin_matrices_from_xml
from model.dualviewunet_simple import UNetDualDecoder
import numpy as np


if __name__ == '__main__':
    out_dir = Path("/media/dl/dataFeb22/results/evaluation")
    test_data_dir = Path("/media/dl/data2/pathlength-reg/datadump/ManualPreprocessed")
    test_data = NPZData(test_data_dir, downsample=2, pad=12, noise=False, binarize=True, eval=True)

    # load model
    model_checkpoint = Path("/media/dl/dataFeb22/results/FUME/iwi5106h/fume-seg-520231/checkpoints/model_99.pth")
    model = UNetDualDecoder(last_activation='sigmoid')
    model.load_state_dict(torch.load(model_checkpoint))
    model.eval()
    model.cuda()

    # instantiate empty container
    selected_name = "Spine06_tulips"
    output = np.zeros((100, 512, 512))

    for x, P, y, fname in test_data:
        if selected_name not in fname.name:
            print(f"done with {selected_name}")
            continue

        # get indices
        parts = fname.name.split("_")
        idx0, idx1 = int(parts[1]), int(parts[2])
        print(idx0)

        y_max = torch.max(y)
        # plt.subplot(141)
        # plt.imshow(x[0].T)
        # plt.subplot(142)
        # plt.imshow(y[0].T, cmap='gray', vmin=0, vmax=y_max)
        # plt.suptitle(fname.name)

        x, P = x.cuda(), P.cuda()
        y_pred = model(torch.unsqueeze(x, dim=0), torch.unsqueeze(P, dim=0))
        y_pred = y_pred.detach().cpu().numpy().squeeze()

        output[idx0 // 4] = y_pred[0].T
        output[idx1 // 4] = y_pred[1].T

        # plt.subplot(143)
        # plt.imshow(y_pred[0].T, cmap='gray', vmin=0, vmax=y_max)
        # plt.subplot(144)
        # plt.imshow(y[0].T - y_pred[0].T, cmap='gray', vmin=0, vmax=y_max)
        # plt.show()

        imsave(out_dir / f"{selected_name}_eval.tiff", output)



# if __name__ == '__main__':
#     testset = Path("/media/dl/dataFeb22/datasets/2022-11-pathlength-regression-fume/simulation/ManualPlacement")
#     test_gt = sorted([f for f in testset.glob("**/*.tiff") if "2Dmask" in f.name])
#     matrices = sorted([f for f in testset.glob("**/*.xml")])
#     test_x = sorted([f for f in testset.glob("**/*.tiff") if "2Dmask" not in f.name])
#

#     assert len(test_gt) == len(test_x), "mismatch in samples and ground truths"
#     for (xpath, Ppath, ypath) in zip(test_x, matrices, test_gt):
#         x, y = imread(xpath), imread(ypath)
#         P = spin_matrices_from_xml(Ppath)
#
#         # create view pairs and
#         for i in range(0, 400 - 180, 4):
#             pass



