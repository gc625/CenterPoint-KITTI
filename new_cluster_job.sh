sbatch -N 1 -n 1 --mem=16000 --nodelist=landonia02 \
-t 04:00:00 \ 
--cpus-per-task=24 \
--gres=gpu:1 \
--output=/home/%u/slurm_logs/slurm-%A_%a.out \
--error=/home/%u/slurm_logs/slurm-%A_%a.out \
cluster_job_submit.sh