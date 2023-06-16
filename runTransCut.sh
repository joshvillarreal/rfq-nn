#!/bin/bash
#SBATCH --job-name=rfq-nn-gpu-notransmissioncut
#SBATCH --output=res_%j.txt
#SBATCH --error=err_%j.txt
#
#SBATCH --time=12:00:00
#SBATCH --mem-per-cpu=64GB
#SBATCH --partition=submit-gpu
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-gpu=1

srun hostname
echo ""
which julia
echo ""
echo "Running Julia Code"
srun --unbuffered julia --project="." -t 2 scan_hyperparameters_withcellnum.jl --data-directory=data/full_with_cellnumber --n-folds=5 --depth-range=4 6 --width-range=50 100 --depth-steps=3 --width-steps=3 --activation-functions=sigmoid --batch-size-range=1024 1024 --batch-size-steps=2 --learning-rate-range=0.001 0.001 --learning-rate-steps=2 --dropout-rate-range=0.0 0.0 --dropout-rate-steps=2 --n-epochs=2500 --log-training-starts --log-training-loss --log-folds --gpu --cut-transmission
echo ""
