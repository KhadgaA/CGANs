#!/bin/bash
#SBATCH --job-name=testing        # Job name
#SBATCH --partition=mtech         # Partition name can be test/small/medium/large/gpu/gpu2 #Partition “gpu or gpu2” should be used only for gpu jobs
#SBATCH --nodes=1               # Run all processes on a single node
#SBATCH --ntasks=1              # Run a single task
#SBATCH --cpus-per-task=4       # Number of CPU cores per task
#SBATCH --gres=gpu:2         # Include gpu for the task (only for GPU jobs)              # Total memory limit (optional)         # Time limit hrs:min:sec (optional)
#SBATCH --output=first_%j.log   # Standard output and error log

module load python/3.10.pytorch
# pip install opencv-python
# python3 hello.py
python3 cGANS.py &> cGAN_run_1.txt &
#CUDA_VISIBLE_DEVICES=2 python main.py --model_name RawGAT_ST --batch_size=24 --language Both --num_epochs 50 --lr 0.000001 --output_dir ./BhashaBluff_Results
#CUDA_VISIBLE_DEVICES=2 python main.py --model_name RawGAT_ST --batch_size 24 --language Both --output_dir ./BhashaBluff_Results
nvidia-smi &

wait