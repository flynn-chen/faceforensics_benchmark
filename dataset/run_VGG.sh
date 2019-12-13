#!/bin/bash
#SBATCH --ntasks 16
#SBATCH -p pi_gerstein
#SBATCH --job-name=DF_VGG
#SBATCH --time=48:00:00
#SBATCH --mem-per-cpu 4G
#SBATCH --cpus-per-task=1
#SBATCH --constraint=avx2
#SBATCH -p pi_gerstein_gpu
#SBATCH --gres=gpu:1

cd /gpfs/ysm/scratch60/gerstein/zc264/FaceForensics/dataset;
module restore cuda
module load miniconda; 
source activate old_keras_gpu_faceforensics; 
python /gpfs/ysm/scratch60/gerstein/zc264/FaceForensics/classification/XceptioNet_keras.py -m VGG -p /gpfs/ysm/scratch60/gerstein/zc264/FaceForensics/dataset/raw/original_sequences/c40/images -n /gpfs/ysm/scratch60/gerstein/zc264/FaceForensics/dataset/raw/manipulated_sequences/Deepfakes/c40/images
