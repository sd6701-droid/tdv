#!/bin/bash
### SLURM ###
#SBATCH --gres=gpu:2
#SBATCH --constraint=h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=24
#SBATCH --mem=480G
#SBATCH --time=47:59:59
#SBATCH --account=torch_pr_230_tandon_priority
#SBATCH --job-name=tdv_small_ssv2_repro
#SBATCH --output=/scratch/sd6701/tdv/logs/slurm/tdv_small_ssv2_repro_%j.log

set -u

export RUN_NAME="tdv_small_ssv2_repro"

# Activate the env explicitly — batch jobs do not inherit an activated env.
source /scratch/sd6701/miniconda3/etc/profile.d/conda.sh
conda activate tdv
python -c "import torch; print('torch', torch.__version__, 'gpus', torch.cuda.device_count())" || exit 1

REPO_ROOT="/scratch/sd6701/tdv"
cd "${REPO_ROOT}" || exit 1
mkdir -p logs/slurm

# ---------------------------------------------------------------------------
# Paper hyperparameters (TDV column), step-based:
# SSv2 ~169k clips; effective batch = 16 clips/GPU x 2 GPUs x 1 accum
# = 32 clips = 512 images (16-frame clips) -- 2x Table C.2's 256.
#   1 epoch ~ 5,279 steps; 20 epochs ~ 105,580 steps; 0.5-epoch warmup ~ 2,640.
# Online eval: linear probe on SSv2 (ImageNet knn needs the gated HF dataset).
# val_check_interval 1.0 -> validation + checkpoint save once per epoch.
# DEVIATION FROM PAPER: batch is 512 images, not Table C.2's 256, traded for
# GPU utilisation (256 images/forward). LR scaled 1e-4 -> 2e-4 to match.
# To revert: batch_size_per_device 8, peak_learning_rate 1e-4,
# max_steps/max_scheduling_steps 211000, warm_up_steps 5300.
# Online probe DISABLED (it idled the GPUs and tripped the cluster's 60%
# utilisation watchdog); run it offline via job_scripts/eval/linear_probe_h100.slurm.
# The probe_eval_* args below are inert until run_online_evaluations is set back to "probe".
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
    --batch_size_per_device 16
    --accumulate_grad_batches 1
    --gradient_clip_val 1.0
    --epochs 20
    --max_steps 105580
    --peak_learning_rate 2e-4
    --min_lr_scale 100000
    --max_scheduling_steps 105580
    --warm_up_steps 2640
    --weight_decay 0.01

    # Dataset
    --dataset_name "aggr"
    --aggr_datasets_list "ssv2"
    --aggr_dataset_dirs "/scratch/sd6701/datasets/ssv2"
    --num_workers 22
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
    --wandb_project "rethinking-jepa"
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
