#!/bin/bash
### SLURM ###
#SBATCH --gres=gpu:2
#SBATCH --constraint=l40s
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=240G
#SBATCH --time=04:00:00
#SBATCH --account=torch_pr_230_tandon_priority
#SBATCH --job-name=tdv_small_l40s_test
#SBATCH --output=/scratch/sd6701/tdv/logs/slurm/tdv_small_l40s_test_%j.log

set -u

export RUN_NAME="tdv_small_l40s_test"

# Activate the env explicitly — batch jobs do not inherit an activated env.
source /scratch/sd6701/miniconda3/etc/profile.d/conda.sh
conda activate tdv

# reduce allocator fragmentation (the OOM run had 4.28 GiB reserved-but-unallocated)
export PYTORCH_ALLOC_CONF=expandable_segments:True
python -c "import torch; print('torch', torch.__version__, 'gpus', torch.cuda.device_count())" || exit 1

REPO_ROOT="/scratch/sd6701/tdv"
cd "${REPO_ROOT}" || exit 1
mkdir -p logs/slurm

# ---------------------------------------------------------------------------
# L40S SMOKE TEST -- validates the pipeline on a smaller/cheaper GPU before
# committing an H100 allocation. Same recipe as pretrain_tdv_h100.slurm.
#
# L40S has 48GB VRAM (vs 80GB on H100), so batch_size_per_device drops 8 -> 4
# with accumulate_grad_batches 2. Effective batch is UNCHANGED at
# 4 clips/GPU x 2 GPUs x 2 accum = 16 clips = 256 images -- still Table C.2.
#
# MEMORY: batch_size_per_device is capped by the iBOT loss, which sends every
# patch token through the 32768-prototype head:
#   bs x 15 frames x 256 patches x 32768 x 4B (fp32 log_softmax)
#   bs=4 -> 1.9 GiB per tensor, ~5 such tensors. bs=8 -> 3.75 GiB (likely OOM on 48GB).
#
# --limit_train_batches 200 makes this finish in minutes. Remove it (and raise
# --time) to run for real. Online probe is disabled; run it via
# job_scripts/eval/linear_probe_h100.slurm.
#
# VERIFY BEFORE SUBMITTING: --constraint, --cpus-per-task and --mem are guesses
# for your L40S partition. Check with:  sinfo -o "%20N %10c %10m %25f %10G"
# ---------------------------------------------------------------------------
ARGS=(
    # Run identity
    --modality "CV"
    --model_name "tdv"
    --run_name ${RUN_NAME}

    # Backbone — ViT-S/14 ("base" for the ViT-B run)
    --backbone_type "dinov2"
    --vit_backbone_size "small"
    --use_dino_head
    --context_length 16
    --time_between_frames 0.25

    # Frame encoder
    --unfreeze_frame_encoder
    --load_without_weights
    --use_ema_for_frame_encoder
    --ema_momentum 0.99

    # Difference encoder
    --change_encoder_type "dinoViT_xattn_small14"
    --difference_encoder_lr_multiplier 3e-2
    --ignore_prefix_tokens_in_condition

    # Loss
    --rollout_n_frames 1
    --use_mse_loss
    --mse_loss_weight 1.5
    --use_dino_loss
    --dino_loss_weight 1.5
    --use_ibot_loss
    --ibot_loss_weight 0.75
    --recon_loss_type "mse"

    # Teacher / centering / sharpening
    --use_centering
    --use_sharpening
    --dino_teacher_temp 0.1
    --dino_student_temp 0.1
    --ibot_teacher_temp 0.1
    --ibot_student_temp 0.1
    --dino_center_update_momentum 0.9
    --ibot_center_update_momentum 0.9
    --dino_head_prototype_dim 32768

    # Hardware
    --gpus "-1"
    --set_matmul_precision "medium"
    --float_precision "bf16-mixed"
    --is_slurm_run

    # Optimiser
    --batch_size_per_device 4
    --accumulate_grad_batches 2
    --gradient_clip_val 1.0
    --epochs 20
    # SMOKE TEST: comment these two out for a real run
    --limit_train_batches 200
    --limit_val_batches 20
    --max_steps 211000
    --peak_learning_rate 1e-4
    --min_lr_scale 100000
    --max_scheduling_steps 211000
    --warm_up_steps 5300
    --weight_decay 0.01

    # Dataset
    --dataset_name "aggr"
    --aggr_datasets_list "ssv2"
    --aggr_dataset_dirs "/scratch/sd6701/datasets/ssv2"
    --num_workers 14
    --image_dim 224 224

    # Validation / checkpointing cadence (checkpoint saved at every validation)
    --val_check_interval 1.0

    # Online evaluation — linear probe on SSv2
    --run_online_evaluations "none"
    --probe_eval_bs 384
    --probe_eval_dataset "ssv2"
    --probe_eval_data_dir "/scratch/sd6701/datasets/ssv2"
    --num_classes 174
    --eval_train_num_samples_per_class -1
    --eval_val_num_samples_per_class -1

    # Wandb
    --wandb_project "tdv"
    --wandb_watch

    # Logging
    --log_var_covar
    --log_model_archi
    --log_every_n_steps 100
    --log_baseline_losses
    --log_gradients

    # Resume: uncomment and point at the run's last.ckpt after a timeout
    # --resume_training_ckpt "logs/checkpoints/${RUN_NAME}_<datetime>_/last.ckpt"
)

# One srun task per GPU — Lightning's SLURM detection requires ntasks == devices.
# Slurm >= 22.05 does not propagate --cpus-per-task from sbatch to srun automatically.
srun --cpus-per-task=${SLURM_CPUS_PER_TASK} python train_model.py "${ARGS[@]}"
