from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model.unet import UNet
from data.loader import NPZData
import argparse
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

def downsample(ten: torch.Tensor):
    return F.pad(ten[:, ::2, ::2], (12, 12, 12, 12), mode="reflect")


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        return 1 - dice


def train_one_epoch(_loader, _model, _loss_fn, _optimizer):
    size = len(_loader.dataset)
    nbatches = len(_loader)
    _model.train()
    avg_loss = 0
    for batch, (x, _, y) in enumerate(_loader):
        # copy to gpu
        x, y = x.cuda(), y.cuda()

        # Compute prediction error
        y_pred = _model(x)
        loss = _loss_fn(y_pred, y)

        # Backpropagation
        _optimizer.zero_grad()
        loss.backward()
        _optimizer.step()
        avg_loss += loss.item()
        writer.add_scalar("loss/train", avg_loss, global_step=e * size + (batch + 1) * _loader.batch_size)

        # every 10% of dataset, print info
        if batch % (nbatches // 10) == 0:
            current = batch * _loader.batch_size
            print(f"{datetime.now()}:   avg loss: {avg_loss / (nbatches // 400):>7f}  [{current:>5d}/{size:>5d}]")
            avg_loss = 0


def evaluate(_loader, _model, _loss_fn):
    print()
    num_batches = len(_loader)
    _model.eval()
    test_loss = 0
    print(f"{datetime.now()}:   evaluation...")
    with torch.no_grad():
        for batch, (x, _, y) in enumerate(_loader):
            # copy to gpu
            x, y = x.cuda(), y.cuda()

            # Compute prediction error
            y_pred = _model(x)
            test_loss += _loss_fn(y_pred, y).item()
            if batch == 0:
                save_prediction_sample(y_pred[0], y[0], x[0])

    test_loss /= num_batches
    writer.add_scalar("loss/test", test_loss, global_step=e * niter)
    print(f"Test Error: \n Avg loss: {test_loss:>8f}")
    return test_loss


def save_prediction_sample(y_pred, y, x):
    y_pred = y_pred.detach().cpu().numpy()
    x = x.detach().cpu().numpy()
    y = y.detach().cpu().numpy()

    # plot image
    fig = plt.figure(figsize=(16, 8))
    fig.suptitle(f"epoch {e}, test loss {test_loss:.2f}")

    plt.subplot(241)
    plt.imshow(x[0].T)
    plt.title("x")
    plt.subplot(245)
    plt.imshow(x[1].T)

    plt.subplot(242)
    plt.imshow(y[0].T)
    plt.title("y")
    plt.subplot(246)
    plt.imshow(y[1].T)

    plt.subplot(243)
    plt.imshow(y_pred[0].T)
    plt.title("y_pred")
    plt.subplot(247)
    plt.imshow(y_pred[1].T)

    plt.subplot(244)
    plt.imshow(y[0].T - y_pred[0].T)
    plt.title("y - y_pred")
    plt.subplot(248)
    plt.imshow(y[1].T - y_pred[1].T)

    fig.tight_layout()
    #plt.show()
    fig.savefig(images_dir / f"example_{e}.png")
    writer.add_figure("example", fig, global_step=e*niter, close=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='dual-view pathlength regression')
    parser.add_argument('--data', action='store', type=Path, required=True)
    parser.add_argument('--testdata', action='store', type=Path, required=True)
    parser.add_argument('--results', action='store', type=Path, required=True)

    parser.add_argument('--epochs', action='store', default=100, type=int, required=False)
    parser.add_argument('--bs', action='store', default=1, type=int, required=False)
    parser.add_argument('--lr', action='store', default=1e-3, type=float, required=False)
    parser.add_argument('--workers', action='store', default=4, type=int, required=False)

    args = parser.parse_args()
    print(args)

    ######## hyper parameters #########
    lr = args.lr
    bs = args.bs
    workers = args.workers
    epochs = args.epochs

    train_data_dir = args.data
    test_data_dir = args.testdata
    checkpoint_dir = Path(args.results / 'checkpoints')
    log_dir = Path(args.results / 'logs')
    images_dir = Path(args.results / 'images')
    ######## hyper parameters #########

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(log_dir=str(log_dir.absolute()))

    # 1. check GPU
    assert torch.cuda.is_available(), "No GPU available. Aborting..."
    print(f"1.\tUsing device {torch.cuda.get_device_name()}")

    # 2. init model
    m = UNet(in_channels=2, out_channels=2, last_activation='sigmoid').cuda()
    model_parameters = filter(lambda p: p.requires_grad, m.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(f"2.\tInitialized model with {np.round(params / 1e6, decimals=2)} mio. params")
    print(f"\t\t- allocated {np.round(torch.cuda.memory_allocated(device='cuda') / 1e6, decimals=2)} Mb")

    # 3. define loss and optimizer
    loss_fn = DiceLoss()
    optimizer = torch.optim.Adam(m.parameters(), lr=lr)

    # 3. init dataloader
    train_data = NPZData(train_data_dir, downsample=2, pad=12, binarize=True, noise=True)  # convert data to (512, 512)
    test_data = NPZData(test_data_dir, downsample=2, pad=12, binarize=True, noise=True)
    nsamples = len(train_data)
    train_loader = DataLoader(train_data,
                              batch_size=bs,
                              pin_memory=True,  # needed for CUDA multiprocessing
                              shuffle=True,
                              num_workers=workers)
    test_loader = DataLoader(test_data,
                             batch_size=bs*2,  # more is possible here bc activations need not be stored
                             pin_memory=True,  # needed for CUDA multiprocessing
                             shuffle=False,
                             num_workers=workers)
    print(f'Number of train samples: {nsamples}')

    # 4. train & save
    test_loss = -1.00
    niter = len(train_data)  # number of iterations per epoch
    for e in range(epochs):
        print(f"Epoch {e + 1}\n-------------------------------")

        # train
        train_one_epoch(train_loader, m, loss_fn, optimizer)

        # log one image and test loss
        test_loss = evaluate(_loader=test_loader, _loss_fn=loss_fn, _model=m)

        # save model params
        torch.save(m.state_dict(), checkpoint_dir / f"model_{e}.pth")
        print(f"Saved Model State to {checkpoint_dir / f'model_{e}.pth'}")
