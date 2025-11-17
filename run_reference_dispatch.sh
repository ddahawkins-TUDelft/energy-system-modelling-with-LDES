#!/bin/bash
#
#SBATCH --job-name="ref_dispatch_test"
#SBATCH --partition=compute
#SBATCH --time=12:00:00          
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6        
#SBATCH --mem-per-cpu=3900         #
#SBATCH --account=research-tpm-ess
#SBATCH --output=logs/%x-%j.out  # logs/ref_dispatch_test-<jobid>.out

#first argument as year range
YEAR_RANGE="$1"

if [ -z "$YEAR_RANGE" ]; then
  echo "No year range provided! Usage: sbatch run_reference_range.sh 2005-2014"
  exit 1
fi

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

echo "Running on host: $(hostname)"
echo "Starting range $YEAR_RANGE at $(date) on $(hostname)"
srun python SoC_proxy_TSA/python/reference_dispatcher.py "$YEAR_RANGE"
echo "Finished range $YEAR_RANGE at $(date)"

