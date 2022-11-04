from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
# from model.unet import UNet
from model.dualviewunet import DualViewUNet
from data.loader import NPZData
import argparse

def train_one_epoch(_loader, _model, _loss_fn, _optimizer):
    size = len(_loader.dataset)
    _model.train()

    for batch, (x, P, y) in enumerate(_loader):
        # copy to gpu
        _model.eval()
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


def evaluate(_loader, _model, _loss_fn):
    size = len(_loader.dataset)
    num_batches = len(_loader)
    _model.eval()
    test_loss = 0
    with torch.no_grad():
        for batch, (x, P, y) in enumerate(_loader):
            # copy to gpu
            x, P, y = x.to(device), P.to(device), y.to(device)

            # Compute prediction error
            y_pred = _model(x, P)
            test_loss += _loss_fn(y_pred, y).item()

    test_loss /= num_batches
    print(f"Test Error: \n Avg loss: {test_loss:>8f}")


def save_prediction_sample(_model, sample, fname):
    sample = np.load(sample)
    with torch.no_grad():
        x = torch.as_tensor(sample['x']).to(device)
        P = torch.as_tensor(sample['P']).to(device)
        y_pred = _model(x, P).numpy()

        # plot image
        fig = plt.figure(figsize=(16, 8))
        plt.subplot(241)
        plt.imshow(sample['x'][0])
        plt.title("x")
        plt.subplot(245)
        plt.imshow(sample['x'][1])

        plt.subplot(242)
        plt.imshow(sample['y'][0])
        plt.title("y")
        plt.subplot(236)
        plt.imshow(sample['y'][1])

        plt.subplot(243)
        plt.imshow(y_pred[0])
        plt.title("y_pred")
        plt.subplot(247)
        plt.imshow(y_pred[1])

        plt.subplot(243)
        plt.imshow(sample['y'][0] - y_pred)
        plt.title("y - y_pred")
        plt.subplot(248)
        plt.imshow(sample['y'][1] - y_pred)

        fig.tight_layout()
        fig.savefig(fname)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='dual-view pathlength regression')
    parser.add_argument('--data', action='store', type=Path, required=True)
    parser.add_argument('--results', action='store', type=Path, required=True)
    parser.add_argument('--example', action='store', type=Path, required=True)

    parser.add_argument('--epochs', action='store', default=100, type=int, required=False)
    parser.add_argument('--bs', action='store', default=1, type=int, required=False)
    parser.add_argument('--lr', action='store', default=1e-3, type=int, required=False)
    parser.add_argument('--workers', action='store', default=4, type=int, required=False)

    args = parser.parse_args()

    ######## hyper parameters #########
    lr = args.lr
    bs = args.bs
    workers = args.workers
    epochs = args.epochs
    example = args.example

    train_data_dir = args.data
    checkpoint_dir = args.results / 'checkpoints'
    images_dir = args.results / 'images'
    ######## hyper parameters #########

    # 1. check GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"1.\tUsing {device} device {torch.cuda.get_device_name() if torch.cuda.is_available() else ''}")

    # 2. init model
    m = DualViewUNet().to(device)
    # m = UNet().to(device)
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
                        pin_memory=True,  # needed for CUDA multiprocessing
                        shuffle=True,
                        num_workers=workers)
    print(f'Number of samples: {nsamples}')

    # 4. train & save
    for e in range(epochs):
        print(f"Epoch {e + 1}\n-------------------------------")
        train_one_epoch(loader, m, loss_fn, optimizer)
        # test_loss = evaluate(_loader=loader, _loss_fn=loss_fn, _model=m)
        save_prediction_sample(_model=m, sample=example, fname=images_dir / f"example_{e}.png")
        torch.save(m.state_dict(), checkpoint_dir / f"model_{e}.pth")
        print("Saved PyTorch Model State to model.pth")
