#!/bin/bash
# Training pipeline for bones_seed dataset
# Step 1: Data preprocessing (run with --limit for testing)
# Step 2: Train MVAE
# Step 3: Train MLD (latent diffusion)

set -e
cd "$(dirname "$0")/.."

DATA_DIR='/media/zxw/WD_BLACK/bones_seed/seq_data_zero_male'

# ── Step 1: Preprocess data ──
# Uncomment --limit 10 for a quick test run
echo "=== Step 1: Preprocessing bones_seed data ==="
python -m data_scripts.extract_dataset_bones_seed \
    --output_dir "$DATA_DIR" \
    # --limit 10

# ── Step 2: Train Motion Primitive VAE ──
echo "=== Step 2: Training MVAE ==="
python -m mld.train_mvae \
    --track 1 \
    --wandb_entity '1419739886-no' \
    --exp_name 'mvae_bones_seed' \
    --data_args.dataset 'mp_seq_v2' \
    --data_args.data_dir "$DATA_DIR" \
    --data_args.cfg_path './config_files/config_hydra/motion_primitive/mp_h2_f8_r8.yaml' \
    --data_args.weight_scheme 'uniform' \
    --train_args.batch_size 128 \
    --train_args.weight_kl 1e-6 \
    --train_args.stage1_steps 100000 \
    --train_args.stage2_steps 50000 \
    --train_args.stage3_steps 50000 \
    --train_args.save_interval 50000 \
    --train_args.weight_smpl_joints_rec 10.0 \
    --train_args.weight_joints_consistency 10.0 \
    --train_args.weight_transl_delta 100 \
    --train_args.weight_joints_delta 100 \
    --train_args.weight_orient_delta 100 \
    --model_args.arch 'all_encoder' \
    --train_args.ema_decay 0.999 \
    --model_args.num_layers 7 \
    --model_args.latent_dim 1 256

# ── Step 3: Train Latent Diffusion Model ──
echo "=== Step 3: Training MLD ==="
python -m mld.train_mld \
    --track 1 \
    --wandb_entity '1419739886-no' \
    --exp_name 'mld_bones_seed' \
    --train_args.batch_size 1024 \
    --train_args.use_amp 1 \
    --data_args.dataset 'mp_seq_v2' \
    --data_args.data_dir "$DATA_DIR" \
    --data_args.cfg_path './config_files/config_hydra/motion_primitive/mp_h2_f8_r4.yaml' \
    --denoiser_args.mvae_path './mvae/mvae_bones_seed/checkpoint_200000.pt' \
    --denoiser_args.train_rollout_type 'full' \
    --denoiser_args.train_rollout_history 'rollout' \
    --train_args.stage1_steps 100000 \
    --train_args.stage2_steps 100000 \
    --train_args.stage3_steps 100000 \
    --train_args.save_interval 100000 \
    --train_args.weight_latent_rec 1.0 \
    --train_args.weight_feature_rec 1.0 \
    --train_args.weight_smpl_joints_rec 0 \
    --train_args.weight_joints_consistency 0 \
    --train_args.weight_transl_delta 1e4 \
    --train_args.weight_joints_delta 1e4 \
    --train_args.weight_orient_delta 1e4 \
    --data_args.weight_scheme 'uniform' \
    denoiser-args.model-args:denoiser-transformer-args
