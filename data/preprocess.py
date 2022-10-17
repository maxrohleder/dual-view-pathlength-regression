from argparse import ArgumentParser, RawDescriptionHelpFormatter
from pathlib import Path

import numpy as np

description = """
This script can be used to convert simulation data generated with ReMAS
to a torch dataset.

Expecting a folder structure as argument <src> like this:

<src>
├───SampleName01
│   ├───<fname>.tiff           <-- make sure <fname> does not include an underscore '_'
│   ├───<fname>_2Dmask.tiff
│   └───<fname>.xml
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
parser.add_argument('--dst', metavar="<dst>", dest='dest', action='store', type=Path, required=True,
                    help='the folder to write the files to.')
parser.add_argument('--dry-run', dest='dry', action='store_true', default=False,
                    help="disables file creation.")
parser.add_argument('--silent', dest='silent', action='store_true', default=False,
                    help="disables status messages.")

args = parser.parse_args()
print(args)
src = Path(args.src)

# 1. select all directories
samplepaths = [f for f in src.iterdir() if f.is_dir()]
skipped = 0


for spath in samplepaths:
    # 2. read images, ground truth and projection matrices
    try:
        assert np.array_equal([f.stem.split("_")[0] for f in spath.iterdir()]), \
            f'there is a non-matching file in {spath}. skipping...'
        fname = spath.iterdir()[0].stem.split("_")[0]

        exit()
    except AssertionError as e:
        print(e)
        skipped += 1
