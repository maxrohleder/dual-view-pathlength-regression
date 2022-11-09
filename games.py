import matplotlib.pyplot as plt
import torch
from tifffile import imsave
from torch.utils.data import DataLoader
import numpy as np


from data.loader import NPZData
from model.dualviewunet_simple import UNetDualDecoder

if __name__ == '__main__':
    # paths
    train_data_dir = r"/media/dl/data2/pathlength-reg/train"

    # params
    bs = 1
    workers = 4

    # init dataloader
    train_data = NPZData(train_data_dir, downsample=2, pad=12)
    nsamples = len(train_data)
    train_loader = DataLoader(train_data,
                              batch_size=bs,
                              pin_memory=True,  # needed for CUDA multiprocessing
                              shuffle=True,
                              num_workers=workers)

    model = UNetDualDecoder()
    model.cuda()
    for i, (x, P, y) in enumerate(train_loader):
        x, P, y = x.cuda(), P.cuda(), y.cuda()
        y_pred = model(y, P)

        y_pred = y_pred.detach().cpu().numpy().squeeze()

        plt.figure(figsize=(8, 4))
        plt.subplot(121)
        plt.suptitle(f"{i}")
        plt.imshow(y.cpu().numpy()[0, 0].T, cmap="Greys")
        plt.imshow(y_pred[0].T, alpha=0.5, cmap="Greens")

        plt.subplot(122)
        plt.imshow(y.cpu().numpy()[0, 1].T, cmap="Greys")
        plt.imshow(y_pred[1].T, alpha=0.5, cmap="Greens")
        plt.tight_layout()

        plt.show()
        base = r"/home/dl/Documents/results/FUME/FUME_on_6_screws_example/examples/"
        imsave(base + r"CM2.tif", y_pred[1].T)
        imsave(base + r"view2.tif", y.cpu().numpy()[0, 1].T)
        input(f"{i}, next?")
