#!/usr/bin/env bash
#SBATCH --cpus-per-task=25
#SBATCH --time=20:00:00
#SBATCH --partition=anonymous-compute
#SBATCH --qos=anonymous-high
#SBATCH --job-name=gendataset
#SBATCH -o ./gendataset.out
#SBATCH -e ./gendataset.out

python -m src.env.robotics.fetch_push
