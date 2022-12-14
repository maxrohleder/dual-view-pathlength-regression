import matplotlib.pyplot as plt
import torch
from tifffile import imsave
from torch import nn
from torch.utils.data import DataLoader
import numpy as np


from data.loader import NPZData
from model.dualviewunet_simple import UNetDualDecoder

if __name__ == '__main__':
    # paths
    train_data_dir = r"/media/dl/data2/pathlength-reg/test"

    # params
    bs = 4
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

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for e in range(100):
        for i, (x, P, y) in enumerate(train_loader):
            x, P, y = x.cuda(), P.cuda(), y.cuda()
            y_pred = model(x, P)

            loss = loss_fn(y_pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f"sample {i} done")

            if i % 10 == 0:
                y_pred = y_pred.detach().cpu().numpy().squeeze()

                plt.figure(figsize=(8, 4))

                plt.subplot(121)
                plt.imshow(y.cpu().numpy()[0, 1].T, cmap="Greys")
                plt.imshow(y_pred[0, 1].T, alpha=0.5, cmap="Greens")

                plt.subplot(122)
                plt.imshow(y.cpu().numpy()[0, 0].T, cmap="Greys")
                plt.imshow(y_pred[0, 0].T, alpha=0.5, cmap="Greens")
                plt.suptitle(f"{i}")
                plt.tight_layout()

                plt.show()
                base = r"/home/dl/Documents/results/FUME/Training-test/"  # adapt this to your needs
                imsave(base + fr"CM1-{e}-{i}.tif", y_pred[0, 0].T)
                imsave(base + fr"view1-{e}-{i}.tif", y.cpu().numpy()[0, 0].T)
                imsave(base + fr"CM2-{e}-{i}.tif", y_pred[0, 1].T)
                imsave(base + fr"view2-{e}-{i}.tif", y.cpu().numpy()[0, 1].T)
                print(f"saved sample {i}")
