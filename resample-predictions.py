from pathlib import Path

import numpy as np
from tifffile import imread, imsave
from tqdm import trange, tqdm

if __name__ == '__main__':
    path = Path("/media/dl/data2/results/fume/UNet_seg_Nov30")
    outpath = path / "whole_scans"
    outpath.mkdir(exist_ok=True)

    # select all predictions
    preds = [f.absolute() for f in path.glob("*.tiff")]

    sample_names_folder = Path("/media/dl/data2/pathlength-reg/ManualPlacement")

    # ['Spine09_6tupils_2rods', 'Spine06', 'Spine07', 'Spine09_6screws_6tulips_2rods_ext', ... , 'Spine01']
    sample_names = [f.name for f in sample_names_folder.iterdir() if f.is_dir()]
    print(sample_names)

    # group into samples
    for s in sample_names:
        selected = [f for f in preds if s in f.name]
        if len(selected) != 55:
            print(f"skipping {s}")
            continue

        out = np.zeros((100, 512, 512))
        score_list = []
        for pair in tqdm(selected, desc=f'resampling {s}'):
            # get indices
            parts = pair.name.split("_")
            idx0, idx1 = int(parts[-3]), int(parts[-2])
            score_list.append(float(parts[1]))

            y_pred = imread(pair)  # (2, 512, 512)

            out[idx0 // 4] = y_pred[0].T
            out[idx1 // 4] = y_pred[1].T

        imsave(outpath / (s + f"_pred_{np.mean(score_list):.4f}.tiff"), out)

        for pair in selected:
            pair.unlink()

