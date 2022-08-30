sbatch -N 1 -n 1 --mem=16000 --nodelist=landonia02 \
--time 04:00:00 \
--cpus-per-task=8 \
--gres=gpu:1 \
--output=/home/%u/slurm_logs/slurm-%A_%a.out \
--error=/home/%u/slurm_logs/slurm-%A_%a.out \
cluster_job_submit.sh