# DART论文解读：基于扩散的自回归运动模型

> **论文标题**: DART: A Diffusion-Based Autoregressive Motion Model for Real-Time Text-Driven Motion Control
> **会议**: ICLR 2025 (Spotlight)
> **代码库**: https://github.com/zkf1997/DART

---

## 一、核心创新

**DART (Diffusion-based Autoregressive motion model for Real-Time control)** 解决了文本驱动人体运动生成的实时性和可控性问题。

### 主要贡献

1. **运动原语VAE** - 将短时运动片段编码到紧凑的潜在空间
2. **潜在空间扩散模型** - 在VAE潜在空间中进行扩散生成，大幅提升速度
3. **自回归生成** - 通过历史帧条件实现无限长度的流畅运动生成
4. **实时控制** - 结合RL策略实现目标导向的运动控制

### 关键优势

- ⚡ **速度**: 比传统扩散模型快10-20倍
- 🎯 **可控**: 支持文本、轨迹、场景约束等多种控制方式
- ♾️ **无限长度**: 自回归生成任意长度的流畅运动
- 🎮 **实时交互**: RL策略实现>300 FPS的运动控制

---

## 二、技术架构

### 整体流程

```
文本提示 → CLIP编码 → 扩散采样 → VAE解码 → SMPLX运动 → 渲染
```

### 三阶段训练

1. **Stage 1: 运动原语VAE训练**
   - 学习运动的紧凑表示
   - 文件: `mld/train_mvae.py`

2. **Stage 2: 潜在扩散模型训练**
   - 在潜在空间学习运动生成
   - 文件: `mld/train_mld.py`

3. **Stage 3: 运动控制策略训练** (可选)
   - 训练RL策略实现目标导向控制
   - 文件: `control/train_reach_location_mld.py`

---

## 三、核心模块详解

### 3.1 运动原语VAE

**论文概念**: 将运动原语 $m \in \mathbb{R}^{T \times D}$ 编码为潜在向量 $z \in \mathbb{R}^{d}$

**代码实现**: `mld/models/architectures/mld_vae.py`

```python
# 架构
class MldVae:
    encoder: Transformer  # 输入: (batch, seq_len, 263)
    decoder: Transformer  # 输出: (batch, seq_len, 263)
    latent_dim: 256      # 潜在向量维度
```

**训练配置**: `mld/train_mvae.py`
- 输入: SMPLX参数序列 (T, 263维)
  - Body pose: 63维
  - Hand pose: 90维
  - Shape: 10维
  - Expression: 10维
  - 其他: 90维
- 输出: 潜在向量 (1, 256)
- 损失函数:
  ```python
  loss = weight_joints_rec * ||joints_pred - joints_gt||²
       + weight_kl * KL(q(z|x) || p(z))
       + weight_joints_delta * ||Δjoints||²
       + weight_transl_delta * ||Δtransl||²
       + weight_orient_delta * ||Δorient||²
  ```

**运动原语配置**: `config_files/config_hydra/motion_primitive/mp_h2_f8_r8.yaml`
- `h2`: 历史帧数 = 2帧
- `f8`: 未来帧数 = 8帧 (运动原语长度)
- `r8`: 重复长度 = 8 (自回归padding)

**关键超参数**:
- Batch size: 128
- 学习率: 1e-4
- KL权重: 1e-6 (逐步退火)
- 训练步数: 200,000

---

### 3.2 潜在扩散模型

**论文概念**: 在潜在空间学习扩散过程 $p_\theta(z_{t-1}|z_t, c)$，其中 $c$ 是文本条件

**代码实现**: `mld/models/architectures/mld_denoiser.py`

```python
class MldDenoiser:
    # 输入
    z_t: 噪声潜在向量 (batch, 1, 256)
    t: 时间步嵌入 (batch, emb_dim)
    c: 文本嵌入 (batch, 512)  # CLIP编码

    # 输出
    ε_θ: 预测的噪声 (batch, 1, 256)
```

**训练过程**: `mld/train_mld.py`
1. 从VAE编码真实运动得到 $z_0$
2. 添加噪声得到 $z_t = \sqrt{\bar{\alpha}_t} z_0 + \sqrt{1-\bar{\alpha}_t} \epsilon$
3. 训练去噪网络预测 $\epsilon$
4. 分类器自由引导: 10%概率丢弃文本条件

**推理加速**: DDIM采样
```python
# 从1000步降至10步
respacing = 'ddim10'
guidance_param = 5.0  # 引导强度
```

**关键超参数**:
- Batch size: 1024
- 学习率: 1e-4
- 扩散步数: 1000
- 训练步数: 300,000
- 混合精度: 启用

---

### 3.3 自回归生成

**论文概念**: 通过历史帧条件实现长序列生成

$$z_i = \text{Denoiser}(z_{i-1}^{\text{repeat}}, c_i)$$

**代码实现**: `mld/rollout_mld.py`, `mld/rollout_demo.py`

```python
# 自回归生成伪代码
full_motion = []
for i in range(num_primitives):
    # 1. 准备历史条件
    history = last_motion[-2:]  # 取最后2帧
    repeat_padding = history[-1].repeat(8)  # 重复最后一帧8次

    # 2. 扩散采样
    latent = denoiser.sample(
        text_embedding=clip_encode(text_prompt),
        history=history,
        guidance_scale=5.0,
        num_steps=10  # DDIM
    )

    # 3. VAE解码
    motion_primitive = vae.decode(latent)  # 8帧新运动

    # 4. 拼接到序列
    full_motion.append(motion_primitive)
```

**实时演示**: `mld/rollout_demo.py`
- 交互式文本输入
- 实时3D可视化 (pyrender)
- 流式SMPLX渲染: `utils/smplx_stream.py`

**使用示例**:
```bash
source ./demos/run_demo.sh
# 输入: "walk forward*5,turn left*3,walk backward*5"
```

---

### 3.4 潜在噪声优化

**论文概念**: 通过优化初始噪声 $z_T$ 满足空间约束

**代码实现**: `mld/optim_mld.py`

```python
# 优化目标
loss = (
    weight_init * ||z_T - z_T^{init}||²        # 保持接近初始噪声
    + weight_floor * floor_penetration_loss     # 脚不穿地
    + weight_jerk * jerk_loss                   # 运动平滑
    + weight_goal * ||joints_final - goal||²    # 到达目标位置
)

# 优化器
optimizer = Adam([z_T], lr=0.05)
for step in range(100):
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

**应用场景**:

1. **运动插值** (`demos/inbetween_babel.sh`)
   - 给定起始和结束关键帧
   - 生成流畅的中间过渡
   - 两种模式:
     - `repeat`: 单起始帧
     - `history`: 多起始帧(保持速度连续)

2. **轨迹控制** (`demos/traj.sh`)
   - 控制手腕或骨盆沿指定轨迹运动
   - 支持稀疏/密集轨迹点
   - 示例: 挥手画圆、走方形路径

3. **场景交互** (`demos/scene.sh`, `mld/optim_scene_mld.py`)
   ```python
   loss += weight_collision * sdf_penetration_loss  # 碰撞约束
   loss += weight_contact * contact_loss            # 接触约束
   loss += weight_skate * foot_sliding_loss         # 防止脚滑
   ```
   - 示例: 爬楼梯、坐椅子
   - 场景SDF计算: `scenes/test_sdf.py`

---

### 3.5 运动控制策略

**论文概念**: 训练RL策略 $\pi(a|s)$ 实时选择潜在动作到达目标

**代码实现**: `control/train_reach_location_mld.py`

**算法**: PPO (Proximal Policy Optimization)

**环境**: `control/env/env_reach_location_mld.py`
```python
# 状态空间
obs = {
    'goal_dir': (2,),           # 目标方向 (x, y)
    'goal_dist': (1,),          # 目标距离
    'goal_angle': (1,),         # 目标角度
    'text_embedding': (512,),   # 技能文本嵌入
    'motion_history': (T, D),   # 运动历史
}

# 动作空间
action = latent_noise  # 连续向量 (latent_dim,)

# 奖励函数
reward = (
    weight_success * success_bonus +
    weight_dist * (-distance_to_goal) +
    weight_foot_floor * (-foot_penetration) +
    weight_skate * (-foot_sliding) +
    weight_orient * (-orientation_error)
)
```

**策略网络**: `control/policy/policy.py`
- 输入: 观察向量
- 输出: 高斯分布参数 (mean, log_std)
- 架构: MLP或Transformer

**训练技巧**:
- 课程学习: 逐步增加目标距离和角度
- 多技能训练: "walk", "run", "hop on left leg"
- 并行环境: 256个环境同时训练

**关键超参数**:
- 环境数: 256
- 学习率: 3e-4
- PPO epochs: 10
- 总时间步: 200M

**使用示例**:
```bash
source ./demos/goal_reach.sh
# 生成走向一系列目标点的运动
```

---

## 四、数据处理流程

### 4.1 数据集准备

**BABEL数据集**: `data_scripts/extract_dataset.py`

1. **输入**:
   - AMASS SMPLX序列 (30 FPS)
   - BABEL文本标注 (JSON格式)

2. **处理流程**:
   ```python
   # 每个序列的BABEL标注格式
   {
       "seq_ann": {
           "labels": [
               {
                   "proc_label": "walk forward",
                   "start_t": 0.0,
                   "end_t": 2.5
               },
               {
                   "proc_label": "turn left",
                   "start_t": 2.5,
                   "end_t": 4.0
               }
           ]
       }
   }
   ```

3. **输出**: 运动原语片段
   ```python
   {
       'motion': (T, 263),      # SMPLX参数
       'text': 'walk forward',  # 文本描述
       'gender': 'male',
       'betas': (10,),          # 体型参数
   }
   ```

**HumanML3D数据集**: `data_scripts/extract_dataset_hml3d_smplh.py`
- 使用SMPL-H身体模型
- 20 FPS (vs BABEL的30 FPS)
- 不同的文本描述风格

### 4.2 数据增强

- 镜像翻转
- 随机起始位置
- 速度扰动

### 4.3 数据加载

**Dataset类**: `mld/data/`
- `HumanML3D.py`: HumanML3D数据集
- `base.py`: 基础数据集类
- `sampling/`: 帧采样策略

---

## 五、推理数据流

```
文本提示 "walk in circles"
    ↓
CLIP编码 → (512,) 文本嵌入
    ↓
扩散采样 (10步DDIM) → 潜在向量 (1, 256)
    ↓
VAE解码 → 运动原语 (8帧, 263维SMPLX参数)
    ↓
SMPLX前向运动学 → 关节位置 (22关节 × 3坐标)
    ↓
渲染器 (pyrender/Blender) → 3D可视化
```

### 可视化工具

**Pyrender查看器**: `visualize/vis_seq.py`
```bash
python -m visualize.vis_seq \
    --add_floor 1 \
    --translate_body 1 \
    --seq_path './output/*.pkl'
```

**Blender渲染**:
1. 导出为NPZ格式
2. 使用SMPL-X Blender插件导入
3. 高质量渲染

---

## 六、性能指标

### 生成速度

| 模块 | 耗时 | FPS |
|------|------|-----|
| VAE编码/解码 | ~1ms | - |
| 扩散采样 (DDIM 10步) | ~50ms | - |
| RL策略推理 | ~3ms | - |
| **自回归生成** | - | **~20 FPS** |
| **RL控制** | - | **>300 FPS** |

### 质量指标 (论文Table 1)

| 指标 | DART | MDM | FlowMDM |
|------|------|-----|---------|
| FID ↓ | **0.15** | 0.32 | 0.21 |
| 多样性 | **9.8** | 9.5 | 9.6 |
| 目标到达精度 | **<10cm** | - | - |

### 对比优势

**相比MDM/MotionDiffuse**:
- ✅ 速度提升10-20倍 (潜在空间扩散)
- ✅ 支持无限长度生成 (自回归)
- ✅ 实时控制能力 (RL策略)

**相比FlowMDM**:
- ✅ 更快的推理速度
- ✅ 更好的运动平滑性
- ⚠️ 需要两阶段训练 (VAE + 扩散)

---

## 七、代码库结构

### 核心文件映射

| 功能模块 | 核心文件 | 作用 |
|---------|---------|------|
| VAE训练 | `mld/train_mvae.py` | 训练运动原语编码器 |
| 扩散训练 | `mld/train_mld.py` | 训练潜在扩散模型 |
| 自回归生成 | `mld/rollout_mld.py` | 长序列运动生成 |
| 实时演示 | `mld/rollout_demo.py` | 交互式文本驱动生成 |
| 噪声优化 | `mld/optim_mld.py` | 约束满足优化 |
| 场景交互 | `mld/optim_scene_mld.py` | 人-场景交互生成 |
| RL训练 | `control/train_reach_location_mld.py` | 目标导向控制策略 |
| 数据处理 | `data_scripts/extract_dataset.py` | BABEL数据预处理 |
| 可视化 | `visualize/vis_seq.py` | 运动序列渲染 |

### 目录结构

```
DART/
├── mld/                    # 核心训练和推理
│   ├── train_mvae.py      # VAE训练
│   ├── train_mld.py       # 扩散模型训练
│   ├── rollout_mld.py     # 自回归生成
│   ├── rollout_demo.py    # 实时演示
│   ├── optim_mld.py       # 噪声优化
│   ├── optim_scene_mld.py # 场景交互
│   ├── models/            # 模型架构
│   └── data/              # 数据加载器
├── control/               # RL控制
│   ├── train_reach_location_mld.py
│   ├── env/               # 环境定义
│   └── policy/            # 策略网络
├── data_scripts/          # 数据处理
│   ├── extract_dataset.py
│   └── extract_dataset_hml3d_smplh.py
├── visualize/             # 可视化工具
│   └── vis_seq.py
├── evaluation/            # 评估脚本
├── demos/                 # 演示脚本
│   ├── run_demo.sh
│   ├── rollout.sh
│   ├── inbetween_babel.sh
│   ├── goal_reach.sh
│   ├── scene.sh
│   └── traj.sh
├── config_files/          # 配置文件
│   └── config_hydra/motion_primitive/
└── utils/                 # 工具函数
    ├── smplx_stream.py
    └── ...
```

---

## 八、使用指南

### 环境配置

```bash
conda env create -f environment.yml
conda activate DART
```

### 下载数据和模型

1. 预训练模型: [Google Drive链接](https://drive.google.com/drive/folders/1vJg3GFVPT6kr6cA0HrQGmiAEBE2dkaps)
2. SMPL-X/H身体模型
3. AMASS数据集 (训练用)
4. BABEL标注 (训练用)

### 快速开始

**1. 交互式文本生成**:
```bash
source ./demos/run_demo.sh
# 输入文本提示，实时生成运动
```

**2. 批量生成**:
```bash
source ./demos/rollout.sh
# 生成"走圆圈"等预定义动作
```

**3. 运动插值**:
```bash
source ./demos/inbetween_babel.sh
# 在两个关键帧间生成过渡
```

**4. 目标到达**:
```bash
source ./demos/goal_reach.sh
# 生成走向目标点的运动
```

**5. 场景交互**:
```bash
source ./demos/scene.sh
# 生成爬楼梯、坐椅子等场景交互
```

### 训练自己的模型

**Step 1: 数据准备**
```bash
python -m data_scripts.extract_dataset
```

**Step 2: 训练VAE**
```bash
python -m mld.train_mvae \
    --track 1 \
    --exp_name 'my_vae' \
    --train_args.batch_size 128 \
    --train_args.stage1_steps 100000
```

**Step 3: 训练扩散模型**
```bash
python -m mld.train_mld \
    --track 1 \
    --exp_name 'my_mld' \
    --denoiser_args.mvae_path './mvae/my_vae/checkpoint_200000.pt' \
    --train_args.batch_size 1024
```

**Step 4: 训练控制策略** (可选)
```bash
python -m control.train_reach_location_mld \
    --track 1 \
    --exp_name 'my_policy' \
    --denoiser_checkpoint './mld_denoiser/my_mld/checkpoint_300000.pt'
```

---

## 九、评估

### 文本条件运动组合
```bash
source ./evaluation/eval_gen_composition.sh
```

### 运动插值
```bash
source ./evaluation/eval_gen_inbetween.sh
```

### 目标到达
```bash
source ./evaluation/eval_gen_goal_reach.sh
```

---

## 十、技术细节

### 运动表示

**SMPLX参数** (263维):
- Body pose: 21关节 × 3 (轴角) = 63维
- Hand pose: 15指关节 × 2手 × 3 = 90维
- Jaw pose: 3维
- Eye pose: 6维
- Shape (β): 10维
- Expression: 10维
- Translation: 3维
- Global orientation: 3维
- 其他: 75维

**关节位置** (66维):
- 22关节 × 3坐标 = 66维

### 扩散过程

**前向过程**:
$$q(z_t|z_0) = \mathcal{N}(z_t; \sqrt{\bar{\alpha}_t}z_0, (1-\bar{\alpha}_t)I)$$

**反向过程**:
$$p_\theta(z_{t-1}|z_t, c) = \mathcal{N}(z_{t-1}; \mu_\theta(z_t, t, c), \Sigma_\theta(z_t, t))$$

**分类器自由引导**:
$$\tilde{\epsilon}_\theta(z_t, c) = \epsilon_\theta(z_t, \emptyset) + s \cdot (\epsilon_\theta(z_t, c) - \epsilon_\theta(z_t, \emptyset))$$

其中 $s$ 是引导强度 (通常为5.0)

### DDIM采样

从1000步降至10步:
```python
# 选择10个时间步
timesteps = [0, 100, 200, ..., 900]

# 每步更新
for t in reversed(timesteps):
    z_t = ddim_step(z_t, t, ε_θ(z_t, t, c))
```

---

## 十一、局限性与未来工作

### 当前局限

1. **两阶段训练**: 需要先训练VAE再训练扩散模型
2. **运动原语长度固定**: 8帧的原语可能不适合所有动作
3. **文本理解**: 依赖CLIP，对复杂语义理解有限
4. **场景交互**: 需要预计算SDF，不支持动态场景

### 未来方向

1. 端到端训练
2. 可变长度运动原语
3. 更强的语言理解 (如GPT)
4. 动态场景交互
5. 多人交互生成

---

## 十二、引用

```bibtex
@inproceedings{Zhao:DartControl:2025,
   title = {{DartControl}: A Diffusion-Based Autoregressive Motion Model for Real-Time Text-Driven Motion Control},
   author = {Zhao, Kaifeng and Li, Gen and Tang, Siyu},
   booktitle = {The Thirteenth International Conference on Learning Representations (ICLR)},
   year = {2025}
}
```

---

## 十三、相关资源

- **项目主页**: https://zkf1997.github.io/DART/
- **论文**: https://arxiv.org/abs/2410.05260
- **代码**: https://github.com/zkf1997/DART
- **联系**: kaifeng.zhao@inf.ethz.ch

---

## 附录：常见问题

### Q1: 如何修改运动原语长度？

修改配置文件 `config_files/config_hydra/motion_primitive/mp_h2_f8_r8.yaml`:
```yaml
future_length: 16  # 从8改为16
```

### Q2: 如何使用自定义文本提示？

BABEL数据集使用动词短语，如:
- "walk forward"
- "turn left"
- "sit down"
- "wave right hand"

查看完整动作列表: https://babel.is.tue.mpg.de/explore.html

### Q3: 如何提高生成速度？

1. 减少DDIM步数: `respacing='ddim5'`
2. 降低引导强度: `guidance_param=3.0`
3. 使用更小的VAE潜在维度

### Q4: 如何训练自定义数据集？

1. 准备SMPLX序列和文本标注
2. 按BABEL格式组织标注
3. 修改 `data_scripts/extract_dataset.py`
4. 重新计算数据统计量 (mean/std)

### Q5: 生成的运动不够平滑怎么办？

增加jerk损失权重:
```python
--train_args.weight_jerk 0.1
```

或在优化时增加平滑约束:
```python
--weight_jerk 0.1
```

---

**文档生成时间**: 2026-04-01
**DART版本**: ICLR 2025 Spotlight
