#!/bin/bash
#SBATCH --job-name="KerasMLP"
#SBATCH --nodes=1
##SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=16GB
##SBATCH --exclusive
#SBATCH -o /OSM/CBR/AF_WQ/source/Franz/Log/Keras/kerasmlp2_%a.txt

module load python/3.6.1
module load keras/2.1.3-py36
module load tensorflow/1.6.0-py36-gpu


python /OSM/CBR/AF_WQ/source/Franz/Keras_many_to_many/MLP/MLP_keras_HPC.py 2000 10 6



