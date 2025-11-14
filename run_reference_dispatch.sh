#!/bin/bash
#
#SBATCH --job-name="ref_dispatch_test"
#SBATCH --partition=compute
#SBATCH --time=08:00:00          
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8        
#SBATCH --mem-per-cpu=3900         #
#SBATCH --account=research-tpm-ess
#SBATCH --output=logs/%x-%j.out  # logs/ref_dispatch_test-<jobid>.out

# Load modules
module load 2025
module load python
module load py-pip
module load gurobi/12.0.0        # adjust if your module version differs

# Activate your venv
source "$HOME/venvs/ldes-env/bin/activate"

# Go to project root
cd "$HOME/projects/energy-system-modelling-with-LDES"

# Just in case: make sure repo root is on PYTHONPATH
export PYTHONPATH="$PWD:$PYTHONPATH"

echo "Starting reference dispatcher test at $(date)"
echo "Running on host: $(hostname)"
echo "Python: $(which python)"
python -V

# Run your script
srun python SoC_proxy_TSA/python/reference_dispatcher.py

echo "Finished at $(date)"

