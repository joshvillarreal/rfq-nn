#!/bin/bash
#SBATCH --job-name=rfq-nn-gpu-test
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
which julia-1
echo ""
echo "Running Julia Code"
srun --unbuffered julia-1 -t 1 scan_hyperparameters.jl --data-directory=data/full_opt_15KeV --n-folds=3 --depth-range=3 3 --width-range=100 100 --depth-steps=2 --width-steps=2 --activation-functions=sigmoid --batch-size-range=1024 1024 --batch-size-steps=2 --learning-rate-range=0.001 0.001 --learning-rate-steps=2 --dropout-rate-range=0.0 0.0 --dropout-rate-steps=2 --n-epochs=50 --log-training-starts=true --log-training-loss=true --log-folds=true --gpu
echo ""
