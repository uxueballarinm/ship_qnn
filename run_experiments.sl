#!/bin/bash
#SBATCH --job-name=qnn_train           # Better name for tracking
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=your.email@domain.com
#SBATCH --time=60:00:00                # Max walltime
#SBATCH --partition=partition_name     # Check with 'sinfo' for valid names
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --mem-per-cpu=600MB
#SBATCH --output=outputs/%x-%j.out
#SBATCH --error=outputs/%x-%j.err

# 1. Create the output directory if it doesn't exist (Slurm won't create it for you)
mkdir -p outputs

# 2. Load and activate environment
ml Miniconda3/23.10.0-1
source ~/.bashrc
conda activate env_name

# 3. Move to your project directory
cd /home/uballarin/ship_qnn_HPC

# 4. Execute (Fixed backslashes to forward slashes)
srun python qnn_train_model.py run \
experiment_definitions/experiments_systematic/bad_angle/3_head_model/option1_good_angle/3_head_model_option_1_experiments_0.yml