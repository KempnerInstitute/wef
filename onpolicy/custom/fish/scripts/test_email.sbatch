#!/bin/bash

#!/bin/bash
#SBATCH -c 1
#SBATCH -p kempner
#SBATCH -t 0:00:30
#SBATCH --account=kempner_krajan_lab
#SBATCH --constraint=cc8.0
#SBATCH --exclude=holygpu8a[19604,19605,19606]
#SBATCH --mem-per-cpu=1G
#SBATCH --gpus-per-node=1
#SBATCH --job-name=email_test
#SBATCH --ntasks=1
#SBATCH --mail-user=satpreet_singh@hms.harvard.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

echo "Job started on $(hostname)"
sleep 5
echo "Job finished at $(date)"

# sbatch -A kempner_krajan_lab test_email.sbatch
