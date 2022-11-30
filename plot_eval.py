import json

import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    dview_seg = "/media/dl/dataFeb22/results/evaluation/Nov30/dual_view_seg/quantitative.json"
    fume_seg = "/media/dl/dataFeb22/results/evaluation/Nov30/fume_seg/quantitative.json"

    dview_seg_quant, fume_quant = None, None
    with open(dview_seg, "r") as f:
        dview_seg_quant = json.load(f)

    with open(fume_seg, "r") as f:
        fume_quant = json.load(f)

    print(len(dview_seg_quant.keys()))
    print(len(fume_quant.keys()))

    dice_fume = [d['recall'] for d in fume_quant.values()]
    dice_2view = [d['recall'] for d in dview_seg_quant.values()]

    print(f"fume: {np.mean(dice_fume)}, pm {np.std(dice_fume)}")
    print(f"unet: {np.mean(dice_2view)}, pm {np.std(dice_2view)}")

    fig, (ax) = plt.subplots(nrows=1, ncols=1, figsize=(2, 4))
    parts = ax.violinplot(
        [dice_fume, dice_2view], showmeans=True, showmedians=True,
        showextrema=False)
    ax.set_xticks(np.arange(1, 3), labels=["fume", "2view"])
    plt.suptitle("Test Set Dice")

    fig = plt.figure()
    recall_fume = [d['recall'] for d in fume_quant.values()]
    precision_fume = [d['precision'] for d in fume_quant.values()]
    plt.scatter(recall_fume, precision_fume)
    recall_unet = [d['recall'] for d in dview_seg_quant.values()]
    precision_unet = [d['precision'] for d in dview_seg_quant.values()]
    plt.scatter(recall_unet, precision_unet)
    plt.show()
