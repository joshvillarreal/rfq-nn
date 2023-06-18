#!/bin/bash
#SBATCH --job-name=rfq-nn-gpu-dropout-d7-lr5
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
srun --unbuffered julia --project="." -t 2 scan_hyperparameters_withcellnum.jl --data-directory=data/full_with_cellnumber --n-folds=5 --depth-range=7 7 --width-range=100 100 --depth-steps=2 --width-steps=2 --activation-functions=sigmoid --batch-size-range=1024 1024 --batch-size-steps=2 --learning-rate-range=0.01 0.01 --learning-rate-steps=2 --dropout-rate-range=0.0 0.25 --dropout-rate-steps=6 --n-epochs=2500 --log-training-starts --log-training-loss --log-folds --gpu --cut-transmission
echo ""
