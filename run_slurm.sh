#!/bin/bash
#SBATCH --job-name=transformer
#SBATCH --mem=128G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --output=%j.log
#SBATCH --partition=
#SBATCH --exclusive

# load the environment
./run.sh

