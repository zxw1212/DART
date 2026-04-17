# Bones Seed 数据集训练 DART

## 文件说明

| 文件 | 用途 |
|------|------|
| `data_scripts/extract_dataset_bones_seed.py` | 数据预处理脚本 |
| `demos/train_bones_seed.sh` | 训练启动脚本（MVAE + MLD） |

## 数据处理逻辑

- 数据源：`/media/zxw/WD_BLACK/bones_seed/smplx_npz/` 
- 文本标注：`metadata/seed_metadata_v002_temporal_labels.jsonl`
- 输出：`/media/zxw/WD_BLACK/bones_seed/seq_data_zero_male/{train,val}.pkl`

处理步骤：
1. 过滤：跳过文件名或文本标签中含有以下关键词的序列（约过滤 16%）：
   `jump, crawl, handstand, fall, roll, lying, faint, postmortem, death, flip, climbing, slide, cartwheel, kneel, floor`
2. gender 强制设为 `male`，betas 置零 `(10,)`
3. 120fps → 30fps（slerp 插值旋转 + 线性插值平移）
4. 计算 joints `[T,22,3]` 和 pelvis_delta `[3]`
5. 文本标注字段映射：`start_time→start_t`, `end_time→end_t`, `description→proc_label`, `act_cat=[]`
6. 随机 90/10 划分 train/val

## 使用命令

```bash
cd /home/zxw/software/DART
source .venv/bin/activate

# 小量测试（10个文件）
python -m data_scripts.extract_dataset_bones_seed --limit 10

# 全量预处理
python -m data_scripts.extract_dataset_bones_seed

# 自定义输出路径
python -m data_scripts.extract_dataset_bones_seed --output_dir /path/to/output
```

## 训练命令

训练参数与原项目 README 一致（先用原始步数，看效果再决定是否加步）：

```bash
# 一键执行全流程（预处理 + MVAE + MLD）
bash demos/train_bones_seed.sh
```

或分步执行：

```bash
# Step 1: MVAE (总步数 200k)
python -m mld.train_mvae --track 1 --wandb_entity '1419739886-no' \
  --exp_name 'mvae_bones_seed' \
  --data_args.dataset 'mp_seq_v2' \
  --data_args.data_dir '/media/zxw/WD_BLACK/bones_seed/seq_data_zero_male' \
  --data_args.cfg_path './config_files/config_hydra/motion_primitive/mp_h2_f8_r8.yaml' \
  --data_args.weight_scheme 'uniform' \
  --train_args.batch_size 128 --train_args.weight_kl 1e-6 \
  --train_args.stage1_steps 100000 --train_args.stage2_steps 50000 --train_args.stage3_steps 50000 \
  --train_args.save_interval 50000 \
  --train_args.weight_smpl_joints_rec 10.0 --train_args.weight_joints_consistency 10.0 \
  --train_args.weight_transl_delta 100 --train_args.weight_joints_delta 100 --train_args.weight_orient_delta 100 \
  --model_args.arch 'all_encoder' --train_args.ema_decay 0.999 \
  --model_args.num_layers 7 --model_args.latent_dim 1 256

如果下次中断时已经有 checkpoint 了，加 --train_args.resume_checkpoint './mvae/mvae_bones_seed/checkpoint_XXXXX.pt' 即可从断点继续

# Step 2: MLD (总步数 300k，需要 MVAE checkpoint)
python -m mld.train_mld --track 1 --wandb_entity '1419739886-no' \
  --exp_name 'mld_bones_seed' \
  --train_args.batch_size 1024 --train_args.use_amp 1 \
  --data_args.dataset 'mp_seq_v2' \
  --data_args.data_dir '/media/zxw/WD_BLACK/bones_seed/seq_data_zero_male' \
  --data_args.cfg_path './config_files/config_hydra/motion_primitive/mp_h2_f8_r4.yaml' \
  --denoiser_args.mvae_path './mvae/mvae_bones_seed/checkpoint_200000.pt' \
  --denoiser_args.train_rollout_type 'full' --denoiser_args.train_rollout_history 'rollout' \
  --train_args.stage1_steps 100000 --train_args.stage2_steps 100000 --train_args.stage3_steps 100000 \
  --train_args.save_interval 100000 \
  --train_args.weight_latent_rec 1.0 --train_args.weight_feature_rec 1.0 \
  --train_args.weight_smpl_joints_rec 0 --train_args.weight_joints_consistency 0 \
  --train_args.weight_transl_delta 1e4 --train_args.weight_joints_delta 1e4 --train_args.weight_orient_delta 1e4 \
  --data_args.weight_scheme 'uniform' \
  denoiser-args.model-args:denoiser-transformer-args
```

## 注意事项

- `weight_scheme` 必须用 `uniform`，因为 bones_seed 没有 `act_cat` 字段，不能用 `text_samp`
- wandb 默认 entity 是原作者的 `interaction`，需要加 `--wandb_entity '1419739886-no'` 指向自己的账号；或用 `--track 0` 关闭 wandb
- MLD 训练依赖 MVAE 的 checkpoint，确保 MVAE 训练完成后再启动
- MVAE cfg 用 `mp_h2_f8_r8.yaml`，MLD cfg 用 `mp_h2_f8_r4.yaml`（与 README 一致）
