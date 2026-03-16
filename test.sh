#!/bin/bash
#SBATCH -C v100-32g
#SBATCH --job-name=test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --time=20:00:00
#SBATCH --account=eko@v100
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ubn25oe@jean-zay.idris.fr
#SBATCH --array=0
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out
#SBATCH --hint=nomultithread
#SBATCH --signal=B:USR1@120
#SBATCH --open-mode=append

echo "==== GPU(s) ===="; nvidia-smi -L
srun hostname
srun echo --------------------------------------

module purge
module load miniforge/24.9.0
module load git
module load cuda/12.8.0
module load cudnn/9.2-v7.5.1.10
module load nccl/2.4.2-1+cuda9.2

conda activate parl


export CUDA_HOME=${CUDA_HOME:-$CUDA_ROOT}
export CUDA_ROOT=${CUDA_ROOT:-$CUDA_HOME}
export PATH=$CUDA_HOME/bin:$PATH

# ✅ cuDNN / libs pip d'abord
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_HOME/lib64

# XLA/JAX
export XLA_FLAGS="--xla_gpu_cuda_data_dir=$CUDA_HOME \
                  --xla_gpu_strict_conv_algorithm_picker=false \
                  --xla_gpu_autotune_level=0"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.8

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF=max_split_size_mb:64,expandable_segments:True
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export WANDB_MODE=offline

# NCCL off (tu as 1 GPU, donc OK)
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,ENV,COLL
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=1

# fork safety
export JAX_DISABLE_MOST_FORKS=1

# sanity check optionnel ici

srun --kill-on-bad-exit=1 -u env -u PYOPENGL_PLATFORM \
  python ./train.py --environment_name=D4RL/pointmaze/umaze-v2 \
    --wandb_experiment_name=ddpm_bc_pointmaze_new \
    --config=./configs/state_config.py:ddpm \
    --num_offline_epochs=3000 --num_online_epochs=0 --seed=0 --num_parallel_envs=1