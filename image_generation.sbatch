#!/bin/bash
#SBATCH -J image_gen        # Job name includes arguments
#SBATCH -o outs/single_image.out    # Output file name
#SBATCH --time=1:00:00     # Time limit
#SBATCH -c 10              # Number of cores
#SBATCH --mem=64G          # Memory limit
#SBATCH -G 1



# Load TensorFlow env
module purge
module load dev/1
module load cuda-11.7.0-gcc-10.3.0-vbhdtgc cudnn-8.2.4.15-11.4-gcc-10.3.0-eluwegp
export XLA_FLAGS="--xla_gpu_cuda_data_dir=$CUDA_HOME"
echo $CUDA_HOME
source ~/.venv/MLEnv3/bin/activate 


# Run the Python script with the two arguments
python3 generateMaxActivations.py 6 3