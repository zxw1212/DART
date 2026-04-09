# BVH 火柴人可视化脚本

对应脚本: `visualize/vis_bvh_stick.py`

这个脚本用于把 `BVH` 文件解析成关节层级和逐帧关节位置，再用 `matplotlib` 以点线火柴人的形式显示骨架动画。

## 功能

- 解析 `BVH` 层级、`OFFSET`、`CHANNELS` 和 `MOTION`
- 根据通道顺序做前向运动学，恢复每一帧的全局关节位置
- 用点和线显示骨架
- 明确表达源坐标系是 `Y-up` 还是 `Z-up`
- 默认把显示坐标统一旋到 `Z-up`
- 支持保持源坐标原样显示
- 支持保存单帧预览图，适合无 GUI 环境

## 坐标系说明

脚本区分两套坐标系:

1. `source-up`
   BVH 文件本身的源坐标系。

2. `display-up`
   可视化时使用的显示坐标系。

默认行为:

- `--source-up auto`
  脚本会优先根据 root 第一个子节点的 `OFFSET` 判断是 `Y-up` 还是 `Z-up`
- `--display-up z`
  无论源坐标是什么，显示时都尽量转成 `Z-up`

如果自动判断为 `Y-up`，显示转到 `Z-up` 时使用如下映射:

```text
X_display = X_source
Y_display = -Z_source
Z_display = Y_source
```

如果想直接看 BVH 原始坐标，不做重定向:

```bash
python3 visualize/vis_bvh_stick.py your_motion.bvh --display-up source
```

## 常用命令

### 1. 直接播放

```bash
python3 visualize/vis_bvh_stick.py /path/to/file.bvh
```

### 2. 指定源坐标系

```bash
python3 visualize/vis_bvh_stick.py /path/to/file.bvh --source-up y
python3 visualize/vis_bvh_stick.py /path/to/file.bvh --source-up z
```

### 3. 保持源坐标原样显示

```bash
python3 visualize/vis_bvh_stick.py /path/to/file.bvh --display-up source
```

### 4. 指定播放区间和抽帧

```bash
python3 visualize/vis_bvh_stick.py /path/to/file.bvh \
  --start-frame 100 \
  --end-frame 400 \
  --stride 2
```

### 5. 保存单帧预览图

这个模式不会弹窗口，适合服务器或终端环境验证:

```bash
python3 visualize/vis_bvh_stick.py /path/to/file.bvh \
  --save-preview /tmp/bvh_preview.png \
  --preview-frame 180
```

## 交互方式

播放窗口内支持以下按键:

- `Space`: 暂停/继续
- `Left`: 后退一帧
- `Right`: 前进一帧
- `q`: 关闭窗口

## 参数说明

- `bvh_path`: 输入 BVH 文件
- `--source-up {auto,y,z}`: 源坐标系 up 轴
- `--display-up {z,source}`: 显示坐标系
- `--start-frame`: 起始帧，0-based
- `--end-frame`: 结束帧，0-based
- `--stride`: 抽帧间隔
- `--fps`: 播放 FPS，默认使用 BVH 的 `Frame Time`
- `--save-preview`: 保存单帧图像并退出
- `--preview-frame`: 保存哪一帧
- `--hide-root-trajectory`: 不显示 root 轨迹

## 测试样例

测试文件:

```text
/home/zxw/Documents/bones_studio_demo/soma_uniform/bvh/210531/jump_and_land_heavy_001__A001.bvh
```

测试命令:

```bash
python3 visualize/vis_bvh_stick.py \
  /home/zxw/Documents/bones_studio_demo/soma_uniform/bvh/210531/jump_and_land_heavy_001__A001.bvh \
  --save-preview /tmp/vis_bvh_stick_preview.png \
  --preview-frame 180
```

这份 BVH 在当前脚本里的自动判断结果是:

```text
source up-axis: Y-up
display orientation: Z-up
axis mapping: X_display = X_source, Y_display = -Z_source, Z_display = Y_source
```

判断依据是 root 第一个子节点偏移中 `Y` 分量明显大于 `Z` 分量。

## 备注

- 脚本依赖 `numpy` 和 `matplotlib`
- 使用 `--save-preview` 时会自动切换到 `Agg` 后端，避免无显示环境报错
- 自动判断 `Y-up/Z-up` 属于启发式逻辑；如果你的 BVH 定义特殊，建议手动指定 `--source-up`
