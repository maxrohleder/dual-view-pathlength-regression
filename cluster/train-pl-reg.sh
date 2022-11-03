#!/bin/bash -l

## Usage:  sbatch train-pl-reg.sh

############    slurm    ###############

#SBATCH --job-name=dual-view-pl-reg
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100
#SBATCH -o /$HOME/%x-%j-on-%N.out
#SBATCH -e /$HOME/%x-%j-on-%N.err

############    paths    ###############

DATA_DIR=$HPCVAULT/pl-reg
FAST_DATA_DIR=$TMPDIR/pl-reg-data-$SLURM_JOB_ID
SRC_DIR=$HOME/dual-view-pathlength-regression
RESULTS_DIR=$HOME/pl-reg-run-$SLURM_JOB_ID
EXAMPLE=$DATA_DIR/Spine02_0_180_id0.npz

############    params   ###############

EPOCHS=100
BS=4
LR=0.001
WORKERS=4

# load and init conda if it doesnt exist
if ! command -v <conda> &> /dev/null 
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
rsync -aq $DATA_DIR/* $FAST_DATA_DIR
echo "finished transfer at $(date)"

# start training
cd $SRC_DIR || echo "could not cd into $SRC_DIR"
python train.py --data $FAST_DATA_DIR --results $RESULTS_DIR --example $EXAMPLE --epochs $EPOCHS --bs $BS --lr $LR --workers $WORKERS

# cleanup
rm -rf $FAST_DATA_DIR