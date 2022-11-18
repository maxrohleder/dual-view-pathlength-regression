#!/bin/bash -l

## Usage:  sbatch train-fume-reg.sh

############    slurm    ###############

#SBATCH --job-name=fume-reg
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100

############    paths    ###############

#DATA_DIR=$HPCVAULT/datadump
DATA_ARCHIVE=$HPCVAULT/pl-reg/archive.tar
FAST_DATA_DIR=$TMPDIR/pl-reg-data-$SLURM_JOB_ID
SRC_DIR=$HOME/dual-view-pathlength-regression
RESULTS_DIR=$HOME/fume-reg-$SLURM_JOB_ID
EXAMPLE=$DATA_DIR/Spine02_0_180_id0.npz

############    params   ###############

EPOCHS=100
BS=4
LR=0.001
WORKERS=4

# allows for internet connection on cluster nodes
export http_proxy=http://proxy:80
export https_proxy=http://proxy:80

# load and init conda if it doesnt exist
if ! command -v conda &> /dev/null
then
	echo "load conda from module"
	module load python/3.8-anaconda
	conda init bash
fi

# activate conda env
conda activate "$HOME/conda/fume"
echo "using python $(which python)"

# check that python can access GPU
nvidia-smi
echo "python reaches gpu: $(python -c 'import torch;print(torch.cuda.is_available())')"

# copy training data to faster drive
echo "started data transfer at $(date)"
#rsync -aq $DATA_DIR/* $FAST_DATA_DIR
mkdir $FAST_DATA_DIR
tar -xf $DATA_ARCHIVE -C $FAST_DATA_DIR
echo "finished transfer at $(date)"

# start training
cd $SRC_DIR || echo "could not cd into $SRC_DIR"
python train_fume_regression.py --data $FAST_DATA_DIR/RandomPreprocessed --testdata $FAST_DATA_DIR/ManualPreprocessed --results $RESULTS_DIR --example $EXAMPLE --epochs $EPOCHS --bs $BS --lr $LR --workers $WORKERS

# cleanup
rm -rf $FAST_DATA_DIR
