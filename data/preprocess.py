from argparse import ArgumentParser, RawDescriptionHelpFormatter
from pathlib import Path

import numpy as np
from tifffile import imread

from data.utils import spin_matrices_from_xml

description = """
This script can be used to convert simulation data generated with ReMAS
to a torch dataset.

Expecting a folder structure as argument <src> like this:

<src>
├───SampleName01
│   ├───<Filename01>.tiff           <-- make sure <Filename01> does not include an underscore '_'
│   ├───<Filename01>_2Dmask.tiff
│   └───<Filename01>.xml
├───SampleName02
...

Will create a dataloader optimized dataset at location <dst> like this:

<dst>
├───SampleName01_3_183_id0.npz
├───SampleName01_4_184_id1.npz
...
├───SampleName02_0_180_id360.npz
├───SampleName02_1_181_id360.npz
...

where each '.npz' file contains a dict like this:

{
    'x': np.ndarray in shape (2, 976, 976)  <-- input x-ray images
    'P': np.ndarray in shape (2, 3, 4)      <-- corresponding projection matrices
    'y': np.ndarray in shape (2, 976, 976)  <-- output path-length maps
}
"""

parser = ArgumentParser(description=description, formatter_class=RawDescriptionHelpFormatter)

parser.add_argument('--src', metavar="<src>", dest='src', action='store', type=Path, required=True,
                    help='the folder to read the simulated data from. see usage for details.')
parser.add_argument('--dst', metavar="<dst>", dest='dst', action='store', type=Path, required=True,
                    help='the folder to write the files to.')
parser.add_argument('--dry-run', dest='dry', action='store_true', default=False,
                    help="disables file creation.")
parser.add_argument('--silent', dest='silent', action='store_true', default=False,
                    help="disables status messages.")

args = parser.parse_args()
src = Path(args.src)
dst = Path(args.dst)

# 1. select all directories
samplepaths = [f for f in src.iterdir() if f.is_dir()]
skipped, processed = 0, 0
sid = 0

for spath in samplepaths:
    # 2. read images, ground truth and projection matrices
    try:
        fname = ''
        print(spath)
        for f in spath.iterdir():
            tmp = f.stem.split('_')[0]
            assert fname == tmp or fname == '', f'there is a non-matching file in {spath}. skipping...'
            fname = tmp
        matrices = spin_matrices_from_xml(spath / (fname + '.xml'))
        assert matrices.shape[0] == 400
        image = imread(spath / (fname + '.tiff'))
        gt2d = imread(spath / (fname + '_2Dmask.tiff'))
        assert matrices.shape[0] == image.shape[0] == gt2d.shape[0] == 400, 'sample doesnt have 400 images'

        for i in range(matrices.shape[0] - 180):
            np.savez(
                dst / (spath.stem + "_" + str(i) + "_" + str(i + 180) + f"_id{sid}"),
                x=np.array([image[i], image[i + 180]]),
                y=np.array([gt2d[i], gt2d[i + 180]]),
                P=np.array([matrices[i], matrices[i + 180]])
            )
            sid += 1

    except AssertionError as e:
        print(e)
        skipped += 1

    except FileNotFoundError as e:
        print(e)
        skipped += 1

    print(f'Done. Overview:'
          f'\n\t- processed sucessfully: {processed}'
          f'\n\t- skipped files: {skipped}'
          f'\n\t- written training pairs: {sid}')
