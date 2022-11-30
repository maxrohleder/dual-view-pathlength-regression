from pathlib import Path

import tqdm
from sklearn.metrics import precision_recall_fscore_support, PrecisionRecallDisplay
import matplotlib.pyplot as plt
from tifffile import imread, imsave
import torch
from torch.utils.data import DataLoader

from data.loader import NPZData
from data.utils import spin_matrices_from_xml
from model.dualviewunet_simple import UNetDualDecoder
from model.unet import UNet
import numpy as np
import json


def seg_stats(y_true, y_pred):
    assert y_true.dtype == bool
    assert y_pred.dtype == bool
    pp = int(np.sum(y_pred))
    p = int(np.sum(y_true))
    tp = int(np.sum(y_pred * y_true))
    tn = int(np.sum(~y_pred * ~y_pred))
    fp = int(np.sum(y_pred * ~y_true))
    fn = int(np.sum(~y_pred * y_true))

    precision = tp / pp
    recall = tp / p
    dice = (2 * tp) / (2 * tp + fp + fn)
    return {'tp': tp, 'tn': tn,'fp': fp, 'pp': pp, 'p': p, 'fn': fn, 'precision': precision, 'recall': recall, 'dice': dice}


def reg_stats(y_true, y_pred):
    pass


def do_eval(_model, _loader, out_dir: Path, needs_p=True, mode='seg'):

    assert mode in ['seg', 'reg'], "only segmentation and regression supported"

    # create output dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # results
    quantitiative = {}

    pbar = tqdm.tqdm(total=len(test_loader), desc='evaluating testset')

    for x, P, y, fname in test_loader:
        x, P = x.cuda(), P.cuda()

        if needs_p:
            y_pred = model(x, P)
        else:
            y_pred = model(x)

        y_pred = y_pred.detach().cpu().numpy().squeeze()
        x = x.cpu().numpy()

        # for every image in batch, get statistics
        for (f, x, y, yp) in zip(fname, x, y, y_pred):
            yp_bin = np.zeros_like(yp, dtype=bool)
            yp_bin[yp > 0.5] = True

            y = y.numpy().astype(bool)

            if mode == 'seg':
                quantitiative[f] = seg_stats(y_true=y, y_pred=yp_bin)
            if mode == 'reg':
                quantitiative[f] = reg_stats(y_true=y, y_pred=y_pred)
                assert False, "NOT IMPLEMENTED"
        pbar.update(1)

    with open(out_dir / 'quantitative.json', 'w', encoding='utf-8') as f:
        json.dump(quantitiative, f, ensure_ascii=False, indent=4)

    return quantitiative



if __name__ == '__main__':
    out_dir = Path("/media/dl/dataFeb22/results/evaluation/Nov30")
    test_data_dir = Path("/media/dl/data2/pathlength-reg/datadump/ManualPreprocessed")
    test_data = NPZData(test_data_dir, downsample=2, pad=12, noise=False, binarize=True, eval=True)
    test_loader = DataLoader(test_data,
                             batch_size=4,
                             pin_memory=True,  # needed for CUDA multiprocessing
                             shuffle=False,
                             num_workers=4)
    # model checkpoints
    fume_seg_checkpoint = Path("/media/dl/dataFeb22/results/FUME/Nov30/fume-seg-522292/checkpoints/model_99.pth")
    fume_reg_checkpoint = Path("/media/dl/dataFeb22/results/FUME/iwi5106h/fume-reg-520145/checkpoints/model_99.pth")
    dual_view_seg_checkpoint = Path("/media/dl/dataFeb22/results/FUME/Nov30/dual-view-seg-522916/checkpoints/model_97_val_dice_0.0675.pth")
    dual_view_reg_checkpoint = Path("/media/dl/dataFeb22/results/FUME/iwi5106h/dual-view-reg-520236/checkpoints/model_99.pth")

    # load segmentation model with fume layers
    model = UNetDualDecoder(last_activation='sigmoid')
    model.load_state_dict(torch.load(fume_seg_checkpoint))
    model.eval()
    model.cuda()
    do_eval(_model=model, _loader=test_loader, out_dir=out_dir / "fume_seg", needs_p=True, mode='seg')

    # load segmentation model
    model = UNet(in_channels=2, out_channels=2, last_activation='sigmoid')
    model.load_state_dict(torch.load(dual_view_seg_checkpoint))
    model.eval()
    model.cuda()
    do_eval(_model=model, _loader=test_loader, out_dir=out_dir / "dual_view_seg", needs_p=False, mode='seg')

    # load regression model with fume layers
    # model = UNetDualDecoder(last_activation='relu')
    # model.load_state_dict(torch.load(fume_reg_checkpoint))
    # model.eval()
    # model.cuda()
    # do_eval(_model=model, _loader=test_loader, out_dir=out_dir / "fume_reg", needs_p=True, mode='reg')

    # load segmentation model
    # model = UNet(in_channels=2, out_channels=2, last_activation='relu')
    # model.load_state_dict(torch.load(dual_view_reg_checkpoint))
    # model.eval()
    # model.cuda()
    # do_eval(_model=model, _loader=test_loader, out_dir=out_dir / "dual_view_reg", needs_p=False, mode='reg')



