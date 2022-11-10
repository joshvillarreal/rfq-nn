#!/bin/bash
#
#SBATCH --job-name=rfq-nn-d6w200lr0.001
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
srun --unbuffered julia -t 2 scan_hyperparameters_temp.jl --data-directory=data/full_opt_15KeV --n-folds=5 --depth-range=6 6 --width-range=200 200 --depth-steps=2 --width-steps=2 --activation-functions=sigmoid --batch-size-range=1024 1024 --batch-size-steps=2 --learning-rate-range=0.001 0.001 --learning-rate-steps=2 --dropout-rate-range=0.0 0.5 --dropout-rate-steps=6 --n-epochs=2500 --log-training-starts=true --log-training-loss=true --log-folds=true --gpu
echo ""
