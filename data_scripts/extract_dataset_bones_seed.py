"""
Preprocess bones_seed dataset for DART training.
Converts SMPL-X NPZ files + temporal text labels into the pkl format expected by WeightedPrimitiveSequenceDatasetV2.

Usage:
    python -m data_scripts.extract_dataset_bones_seed [--limit N] [--output_dir PATH]
"""
import argparse
import json
import pickle
import random
from pathlib import Path

import numpy as np
import torch
from pytorch3d import transforms
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from tqdm import tqdm

from utils.smpl_utils import PrimitiveUtility

# ── constants ──
BONES_SEED_ROOT = Path('/media/zxw/WD_BLACK/bones_seed')
NPZ_DIR = BONES_SEED_ROOT / 'smplx_npz'
LABELS_PATH = BONES_SEED_ROOT / 'metadata' / 'seed_metadata_v002_temporal_labels.jsonl'
DEFAULT_OUTPUT_DIR = BONES_SEED_ROOT / 'seq_data_zero_male'
TARGET_FPS = 30
TRAIN_RATIO = 0.9

SKIP_KEYWORDS = [
    "jump", "crawl", "handstand", "fall", "roll", "lying", "faint",
    "postmortem", "death", "flip", "climbing", "slide", "cartwheel",
    "kneel", "floor",
]

device = 'cuda'
primitive_utility = PrimitiveUtility(device=device)


def should_skip(filename, events=None):
    """Check if a sequence should be skipped based on filename or text labels."""
    name_lower = filename.lower()
    for kw in SKIP_KEYWORDS:
        if kw in name_lower:
            return True
    if events:
        for evt in events:
            desc_lower = evt['description'].lower()
            for kw in SKIP_KEYWORDS:
                if kw in desc_lower:
                    return True
    return False


# ── reuse from extract_dataset.py ──
def downsample(fps, target_fps, seq_data):
    old_trans = seq_data['trans']
    old_poses = seq_data['poses'][:, :66].reshape((-1, 22, 3))
    old_num_frames = len(seq_data['trans'])
    new_num_frames = int((old_num_frames - 1) / fps * target_fps) + 1
    if new_num_frames < 2:
        return None, None
    old_time = np.array(range(old_num_frames)) / fps
    new_time = np.array(range(new_num_frames)) / target_fps
    trans = np.zeros((new_num_frames, 3))
    poses = np.zeros((new_num_frames, 22, 3))
    for i in range(3):
        trans[:, i] = np.interp(x=new_time, xp=old_time, fp=old_trans[:, i])
    for joint_idx in range(22):
        slerp = Slerp(times=old_time, rotations=R.from_rotvec(old_poses[:, joint_idx, :]))
        poses[:, joint_idx, :] = slerp(new_time).as_rotvec()
    return trans, poses.reshape((-1, 66))


def calc_joints_pelvis_delta(motion_data):
    pelvis_delta = primitive_utility.calc_calibrate_offset({
        'betas': torch.tensor(motion_data['betas'], device=device).reshape(1, 10),
        'gender': motion_data['gender'],
    })  # [1, 3]
    pelvis_delta = pelvis_delta.detach().cpu().numpy().squeeze()  # [3]
    num_frames = len(motion_data['trans'])
    poses = torch.tensor(motion_data['poses'], device=device)
    global_orient = transforms.axis_angle_to_matrix(poses[:, :3])
    body_pose = transforms.axis_angle_to_matrix(poses[:, 3:66].reshape(num_frames, 21, 3))
    joints = primitive_utility.smpl_dict_inference(
        {
            'gender': motion_data['gender'],
            'betas': torch.tensor(motion_data['betas'], device=device).reshape(1, 10).repeat(num_frames, 1),
            'transl': torch.tensor(motion_data['trans'], device=device).reshape(num_frames, 3),
            'global_orient': global_orient,
            'body_pose': body_pose,
        }, return_vertices=False
    )  # [num_frames, 22, 3]
    joints = joints.detach().cpu().numpy()
    return joints, pelvis_delta


# ── load temporal labels ──
def load_temporal_labels(labels_path):
    """Returns dict: filename -> list of events"""
    mapping = {}
    with open(labels_path, 'r') as f:
        for line in f:
            entry = json.loads(line.strip())
            mapping[entry['filename']] = entry['events']
    return mapping


# ── main processing ──
def process_bones_seed(limit=None, output_dir=None):
    output_dir = Path(output_dir) if output_dir else DEFAULT_OUTPUT_DIR
    output_dir.mkdir(exist_ok=True, parents=True)

    # load temporal labels
    label_map = load_temporal_labels(LABELS_PATH)
    print(f'Loaded temporal labels for {len(label_map)} sequences')

    # collect all npz files
    npz_files = sorted(NPZ_DIR.rglob('*.npz'))
    print(f'Found {len(npz_files)} npz files')
    if limit:
        npz_files = npz_files[:limit]
        print(f'Processing first {limit} files only')

    all_data = []
    skipped = 0
    skipped_keyword = 0

    for npz_path in tqdm(npz_files, desc='Processing'):
        seq_name = npz_path.stem  # e.g. "jump_and_land_heavy_001__A001"

        # skip sequences matching banned keywords in filename or text labels
        events = label_map.get(seq_name)
        if should_skip(seq_name, events):
            skipped_keyword += 1
            continue

        seq_data = dict(np.load(npz_path, allow_pickle=True))

        fps = float(seq_data['fps'])

        # force gender=male, betas=zeros(10)
        motion_data = {}
        motion_data['gender'] = 'male'
        motion_data['betas'] = np.zeros(10, dtype=np.float32)

        # downsample from ~120fps to 30fps
        # downsample() expects seq_data with 'trans' and 'poses' keys
        ds_input = {
            'trans': seq_data['trans'].astype(np.float32),
            'poses': seq_data['poses'].astype(np.float32),
        }
        trans, poses = downsample(fps, TARGET_FPS, ds_input)
        if trans is None:
            skipped += 1
            continue
        motion_data['trans'] = trans.astype(np.float32)
        motion_data['poses'] = poses.astype(np.float32)

        # compute joints and pelvis_delta
        joints, pelvis_delta = calc_joints_pelvis_delta(motion_data)
        motion_data['joints'] = joints
        motion_data['pelvis_delta'] = pelvis_delta

        seq_data_dict = {
            'motion': motion_data,
            'data_source': 'bones_seed',
            'seq_name': str(npz_path.relative_to(NPZ_DIR)),
        }

        # map temporal labels
        if seq_name in label_map:
            frame_labels = []
            for evt in label_map[seq_name]:
                frame_labels.append({
                    'start_t': evt['start_time'],
                    'end_t': evt['end_time'],
                    'proc_label': evt['description'],
                    'act_cat': [],
                })
            seq_data_dict['frame_labels'] = frame_labels
        else:
            # fallback: single label covering entire sequence
            duration = motion_data['trans'].shape[0] / TARGET_FPS
            seq_data_dict['frame_labels'] = [{
                'start_t': 0,
                'end_t': duration,
                'proc_label': 'a person performs an action',
                'act_cat': [],
            }]

        all_data.append(seq_data_dict)

    print(f'Processed {len(all_data)} sequences, skipped {skipped} (too short), skipped {skipped_keyword} (keyword filter)')

    # train/val split
    random.seed(42)
    random.shuffle(all_data)
    split_idx = int(len(all_data) * TRAIN_RATIO)
    train_data = all_data[:split_idx]
    val_data = all_data[split_idx:]
    print(f'Train: {len(train_data)}, Val: {len(val_data)}')

    with open(output_dir / 'train.pkl', 'wb') as f:
        pickle.dump(train_data, f)
    with open(output_dir / 'val.pkl', 'wb') as f:
        pickle.dump(val_data, f)
    print(f'Saved to {output_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=None, help='Process only first N files (for testing)')
    parser.add_argument('--output_dir', type=str, default=None)
    args = parser.parse_args()
    process_bones_seed(limit=args.limit, output_dir=args.output_dir)
