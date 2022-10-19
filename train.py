from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model.unet import UNet
from data.loader import NPZData


def train_one_epoch(_loader, _model, _loss_fn, _optimizer):
    size = len(_loader.dataset)
    _model.train()

    for batch, (x, P, y) in enumerate(_loader):
        # copy to gpu
        x, P, y = x.to(device), P.to(device), y.to(device)

        # Compute prediction error
        y_pred = _model(x, P)
        loss = _loss_fn(y_pred, y)

        # Backpropagation
        _optimizer.zero_grad()
        loss.backward()
        _optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(x)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(_loader, _model, _loss_fn):
    size = len(_loader.dataset)
    num_batches = len(_loader)
    _model.eval()
    test_loss = 0
    with torch.no_grad():
        for X, y in _loader:
            X, y = X.to(device), y.to(device)
            pred = _model(X)
            test_loss += _loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


if __name__ == '__main__':
    ######## hyper parameters #########
    lr = 1e-3
    bs = 16
    workers = 4
    epochs = 100
    checkpoint_dir = ''
    train_data_dir = Path(r'D:\datasets\2022-10-projection-domain-segmentation\simulation')
    ######## hyper parameters #########

    # 1. check GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"1.\tUsing {device} device {torch.cuda.get_device_name() if torch.cuda.is_available() else ''}")

    # 2. init model
    m = UNet().to(device)
    model_parameters = filter(lambda p: p.requires_grad, m.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(f"2.\tInitialized model with {np.round(params / 1e6, decimals=2)} mio. params")
    print(f"\t\t- allocated {np.round(torch.cuda.memory_allocated(device=device) / 1e6, decimals=2)} Mb")

    # 3. define loss and optimizer
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(m.parameters(), lr=lr)

    # 3. init dataloader
    data = NPZData(train_data_dir)
    nsamples = len(data)
    loader = DataLoader(data,
                        batch_size=bs,
                        pin_memory=True,  # needed for CUDA multiproc
                        shuffle=True,
                        num_workers=workers)
    print(f'Number of samples: {nsamples}')

    for e in range(epochs):
        print(f"Epoch {e + 1}\n-------------------------------")
        train_one_epoch(loader, m, loss_fn, optimizer)
        test_loss = test()
        torch.save(m.state_dict(), checkpoint_dir / f"model_{test_loss:.2f}_{e}.pth")
        print("Saved PyTorch Model State to model.pth")

    # print(x.shape)
    # x = x.numpy()[0]
    # y = y.numpy()[0]
    # plt.subplot(221)
    # plt.imshow(x[0])
    # plt.subplot(222)
    # plt.imshow(x[1])
    # plt.subplot(223)
    # plt.imshow(y[0])
    # plt.subplot(224)
    # plt.imshow(y[1])
    # plt.show()
