source /conda/etc/profile.d/conda.sh
conda activate rapids

echo "Running: train.py $@"
python train.py $@
