python download-Faceforensics.py /gpfs/ysm/scratch60/gerstein/zc264/FaceForensics/dataset/raw/ -c c40 -t videos -d original --num_videos 250 --server CA
python download-Faceforensics.py /gpfs/ysm/scratch60/gerstein/zc264/FaceForensics/dataset/raw/original_masks/ -c c40 -t masks -d original --num_videos 250 --server CA

python download-Faceforensics.py /gpfs/ysm/scratch60/gerstein/zc264/FaceForensics/dataset/raw/ -c c40 -t videos -d Deepfakes --num_videos 250 --server CA
python download-Faceforensics.py /gpfs/ysm/scratch60/gerstein/zc264/FaceForensics/dataset/raw/ -c c40 -t masks -d Deepfakes --num_videos 250 --server CA

python download-Faceforensics.py /gpfs/ysm/scratch60/gerstein/zc264/FaceForensics/dataset/raw/ -c c40 -t videos -d NeuralTextures --num_videos 250 --server CA
python download-Faceforensics.py /gpfs/ysm/scratch60/gerstein/zc264/FaceForensics/dataset/raw/ -c c40 -t masks -d NeuralTextures --num_videos 250 --server CA
