#!/bin/bash
#SBATCH -J image_gen_${ARGS[$SLURM_ARRAY_TASK_ID]}        # Job name includes arguments
#SBATCH -o outs/image_gen_%a.out    # Output file name
#SBATCH --time=18:00:00     # Time limit
#SBATCH -c 10              # Number of cores
#SBATCH --mem=64G          # Memory limit
#SBATCH -G 1
#SBATCH --array=0-5        # Number of jobs to run

# Read in the arguments from a file
ARGS=($(cat args.txt))

# Load TensorFlow env
module purge
module load dev/1
module load cuda-11.7.0-gcc-10.3.0-vbhdtgc cudnn-8.2.4.15-11.4-gcc-10.3.0-eluwegp
export XLA_FLAGS="--xla_gpu_cuda_data_dir=$CUDA_HOME"
echo $CUDA_HOME
source ~/.venv/MLEnv3/bin/activate 

# Get the arguments for the current job index
INDEX=$((SLURM_ARRAY_TASK_ID * 2))
ARG1=${ARGS[$INDEX]}
ARG2=${ARGS[$INDEX+1]}

# Run the Python script with the two arguments
python3 generateMaxActivations.py $ARG1 $ARG2