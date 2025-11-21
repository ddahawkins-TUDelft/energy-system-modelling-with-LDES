#!/bin/bash
#
#SBATCH --job-name="cluster_dispatch"
#SBATCH --partition=compute
#SBATCH --time=08:00:00          
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=3000         
#SBATCH --account=research-tpm-ess
#SBATCH --output=logs/%x-%j.out 



# Load modules
module load miniconda3
module load slurm
module load gurobi/12.0.0    # adjust version if needed

# --- Activate your Python virtual environment ---
conda activate /scratch/$USER/conda/envs/py312

# Go to project root
cd "$HOME/projects/energy-system-modelling-with-LDES"

# Just in case: make sure repo root is on PYTHONPATH
export PYTHONPATH="$PWD:$PYTHONPATH"

echo "Running on host: $(hostname)"
srun python SoC_proxy_TSA/python/post_optimised_cluster_tsa_dispatcher.py 

