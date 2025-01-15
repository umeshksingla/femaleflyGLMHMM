#!/bin/bash
#SBATCH --time=1:10:00
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --output='/home/us3519/femaleflylogs/sweep_hyp.%A.%a.log'
#SBATCH --mail-type=begin
#SBATCH --mail-type=fail
#SBATCH --mail-type=end

PY_SCRIPT="crossvalidation/run_single.py"
echo "$PY_SCRIPT"

array_args_file="$1"
echo "$array_args_file"

linenum=$SLURM_ARRAY_TASK_ID
echo "SLURM_ARRAY_TASK_ID: $linenum"
model_config=$(sed -n "$linenum p" $array_args_file)

#source activate keypoint_moseq_new
export XLA_PYTHON_CLIENT_PREALLOCATE=false
python "$PY_SCRIPT" -mc "$model_config"
