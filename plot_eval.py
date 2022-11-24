import json

import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    dview_seg = "/media/dl/dataFeb22/results/evaluation/dual_view_seg/quantitative.json"
    fume_seg = "/media/dl/dataFeb22/results/evaluation/fume_seg/quantitative.json"

    dview_seg_quant, fume_quant = None, None
    with open(dview_seg, "r") as f:
        dview_seg_quant = json.load(f)

    with open(fume_seg, "r") as f:
        fume_quant = json.load(f)

    print(len(dview_seg_quant.keys()))
    print(len(fume_quant.keys()))

    dice_fume = [d['dice'] for d in fume_quant.values()]
    dice_2view = [d['dice'] for d in dview_seg_quant.values()]

    fig, (ax) = plt.subplots(nrows=1, ncols=1, figsize=(2, 4))
    parts = ax.violinplot(
        [dice_fume, dice_2view], showmeans=True, showmedians=True,
        showextrema=False)
    ax.set_xticks(np.arange(1, 3), labels=["fume", "2view"])
    plt.suptitle("Test Set Dice")
    plt.show()
