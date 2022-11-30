import numpy as np
from pathlib import Path
import argparse

import torch
from tifffile import imsave
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.loader import NPZData
from model.dualviewunet_simple import UNetDualDecoder
from model.unet import UNet


def dice(y_true, y_pred, smooth=1e-5):
    # flatten label and prediction tensors
    y_pred, y_true = y_pred.flatten(), y_true.flatten()

    intersection = (y_true * y_pred).sum()
    dice = (2. * intersection + smooth) / (y_pred.sum() + y_true.sum() + smooth)

    return dice


def mean_abs_error(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def forground_mean_abs_error(y_true, y_pred):
    y_true, y_pred = y_true.flatten(), y_pred.flatten()
    y_pred_masked = y_pred[y_true > 0]
    y_true_masked = y_true[y_true > 0]
    return np.mean(np.abs(y_true_masked - y_pred_masked))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='dual-view pathlength regression')
    parser.add_argument('--data', action='store', type=Path, required=True)
    parser.add_argument('--dst', action='store', type=Path, required=True)
    parser.add_argument('--model', action='store', type=str, required=True,
                        choices=['UNet_seg', 'UNet_reg', 'UNet_fume_seg', 'UNet_fume_reg'])
    parser.add_argument('--checkpoint', action='store', type=Path, required=True)

    parser.add_argument('--bs', action='store', default=1, type=int, required=False)
    parser.add_argument('--workers', action='store', default=4, type=int, required=False)

    # --data
    # /media/dl/data2/pathlength-reg/datadump/ManualPreprocessed
    # --dst
    # /media/dl/data2/results/fume/FUME_seg_Nov30
    # --model
    # UNet_fume_seg
    # --checkpoint
    # /media/dl/dataFeb22/results/FUME/iwi5106h/fume-seg-520231/checkpoints/model_99.pth

    # --data
    # /media/dl/data2/pathlength-reg/datadump/ManualPreprocessed
    # --dst
    # /media/dl/data2/results/fume/UNet_seg_Nov30
    # --model
    # UNet_seg
    # --checkpoint
    # /media/dl/dataFeb22/results/FUME/Nov30/dual-view-seg-522916/checkpoints/model_97_val_dice_0.0675.pth

    args = parser.parse_args()

    # instantiate right model
    if args.model == 'UNet_seg':
        model = UNet(in_channels=2, out_channels=2, last_activation='sigmoid')
    elif args.model == 'UNet_reg':
        model = UNet(in_channels=2, out_channels=2, last_activation='relu')
    elif args.model == 'UNet_fume_seg':
        model = UNetDualDecoder(last_activation='sigmoid')
    else:
        model = UNetDualDecoder(last_activation='relu')

    model.load_state_dict(torch.load(args.checkpoint))
    model.eval()
    model.cuda()

    # instantiate a dataset and dataloader with test data
    is_seg = True if 'seg' in args.model else False
    needs_p = True if 'fume' in args.model else False
    test_data = NPZData(args.data, downsample=2, pad=12, noise=True, binarize=is_seg, eval=True)
    test_loader = DataLoader(test_data,
                             batch_size=4,
                             pin_memory=True,  # needed for CUDA multiprocessing
                             shuffle=False,
                             num_workers=4)

    # predict all batches and save each sample into outputfolder
    pbar = tqdm(total=len(test_loader), desc='evaluating testset')

    score_list = []
    for x, P, y, fname in test_loader:
        x, P = x.cuda(), P.cuda()

        if needs_p:
            y_pred = model(x, P)
        else:
            y_pred = model(x)

        y_pred = y_pred.detach().cpu().numpy().squeeze()

        # for every image in batch, get statistics and save
        for (f, y, yp) in zip(fname, y, y_pred):
            f = Path(f)

            if is_seg:
                y = y.numpy().astype(bool)
                yp_bin = np.zeros_like(yp, dtype=bool)
                yp_bin[yp > 0.5] = True
                score = dice(y_true=y, y_pred=yp_bin)
            else:
                score = forground_mean_abs_error(y_true=y.numpy(), y_pred=yp)

            score_list.append(float(score))
            imsave(args.dst / (f"s_{score:.4f}_" + f.stem + '.tiff'), yp)
        pbar.update(1)
        pbar.set_description(f"avg score: {np.mean(score_list):.4f}")
