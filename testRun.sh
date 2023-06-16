#!/bin/bash
#SBATCH --job-name=rfq-nn-gpu-cuttransmission-test
#SBATCH --output=res_%j.txt
#SBATCH --error=err_%j.txt
#
#SBATCH --time=12:00:00
#SBATCH --mem-per-cpu=64GB
#SBATCH --partition=submit-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=1

srun hostname
echo ""
which julia
echo ""
echo "Running Julia Code"
srun --unbuffered julia --project="." -t 3 scan_hyperparameters_withcellnum.jl --data-directory=data/full_with_cellnumber --n-folds=2 --depth-range=3 5 --width-range=50 50 --depth-steps=3 --width-steps=2 --activation-functions=sigmoid --batch-size-range=1024 1024 --batch-size-steps=2 --learning-rate-range=0.001 0.001 --learning-rate-steps=2 --dropout-rate-range=0.0 0.0 --dropout-rate-steps=2 --n-epochs=25 --log-training-starts --log-training-loss --log-folds --gpu --cut-transmission
echo ""
