#!/usr/bin/env python3
"""BVH stick-figure visualizer with explicit Y-up/Z-up handling."""

import argparse
import os
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

import matplotlib

if "--save-preview" in sys.argv:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


AXIS_INDEX = {"X": 0, "Y": 1, "Z": 2}


@dataclass
class Joint:
    name: str
    parent: int
    offset: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    channels: List[str] = field(default_factory=list)
    children: List[int] = field(default_factory=list)
    is_end_site: bool = False


@dataclass
class BvhData:
    joints: List[Joint]
    motion: np.ndarray
    frame_time: float

    @property
    def num_frames(self) -> int:
        return int(self.motion.shape[0])

    @property
    def num_channels(self) -> int:
        return int(self.motion.shape[1])


def parse_bvh(path: Path) -> BvhData:
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    joints: List[Joint] = []
    stack: List[int] = []
    motion_line_idx = None
    idx = 0

    while idx < len(lines):
        line = lines[idx].strip()
        idx += 1
        if not line or line == "HIERARCHY":
            continue
        if line == "MOTION":
            motion_line_idx = idx - 1
            break
        if line.startswith(("ROOT ", "JOINT ")):
            _, name = line.split(maxsplit=1)
            parent = stack[-1] if stack else -1
            joint_idx = len(joints)
            joints.append(Joint(name=name, parent=parent))
            if parent >= 0:
                joints[parent].children.append(joint_idx)
            stack.append(joint_idx)
            continue
        if line == "End Site":
            if not stack:
                raise ValueError("Encountered 'End Site' without a parent joint.")
            parent = stack[-1]
            end_idx = len(joints)
            joints.append(
                Joint(
                    name=f"{joints[parent].name}_EndSite",
                    parent=parent,
                    is_end_site=True,
                )
            )
            joints[parent].children.append(end_idx)
            stack.append(end_idx)
            continue
        if line == "{":
            continue
        if line == "}":
            if not stack:
                raise ValueError("Unbalanced closing brace in BVH hierarchy.")
            stack.pop()
            continue
        if line.startswith("OFFSET "):
            if not stack:
                raise ValueError("Encountered OFFSET outside a joint scope.")
            parts = line.split()
            joints[stack[-1]].offset = np.asarray(parts[1:4], dtype=np.float64)
            continue
        if line.startswith("CHANNELS "):
            if not stack:
                raise ValueError("Encountered CHANNELS outside a joint scope.")
            parts = line.split()
            num_channels = int(parts[1])
            joints[stack[-1]].channels = parts[2 : 2 + num_channels]
            continue
        raise ValueError(f"Unsupported BVH line: {line}")

    if motion_line_idx is None:
        raise ValueError(f"No MOTION section found in {path}.")
    if not joints:
        raise ValueError(f"No joints parsed from {path}.")

    frames_line = lines[motion_line_idx + 1].strip()
    frame_time_line = lines[motion_line_idx + 2].strip()
    if not frames_line.startswith("Frames:"):
        raise ValueError(f"Invalid BVH frame count line: {frames_line}")
    if not frame_time_line.startswith("Frame Time:"):
        raise ValueError(f"Invalid BVH frame time line: {frame_time_line}")

    num_frames = int(frames_line.split(":", 1)[1].strip())
    frame_time = float(frame_time_line.split(":", 1)[1].strip())
    num_channels = sum(len(joint.channels) for joint in joints)

    motion_text = " ".join(line.strip() for line in lines[motion_line_idx + 3 :] if line.strip())
    motion_values = np.fromstring(motion_text, sep=" ", dtype=np.float64)
    expected_values = num_frames * num_channels
    if motion_values.size != expected_values:
        raise ValueError(
            f"Motion channel count mismatch: expected {expected_values}, got {motion_values.size}."
        )
    motion = motion_values.reshape(num_frames, num_channels)
    return BvhData(joints=joints, motion=motion, frame_time=frame_time)


def rotation_matrix_batch(axis: str, angles_rad: np.ndarray) -> np.ndarray:
    c = np.cos(angles_rad)
    s = np.sin(angles_rad)
    batch = angles_rad.shape[0]
    rot = np.zeros((batch, 3, 3), dtype=np.float64)

    if axis == "X":
        rot[:, 0, 0] = 1.0
        rot[:, 1, 1] = c
        rot[:, 1, 2] = -s
        rot[:, 2, 1] = s
        rot[:, 2, 2] = c
        return rot
    if axis == "Y":
        rot[:, 0, 0] = c
        rot[:, 0, 2] = s
        rot[:, 1, 1] = 1.0
        rot[:, 2, 0] = -s
        rot[:, 2, 2] = c
        return rot
    if axis == "Z":
        rot[:, 0, 0] = c
        rot[:, 0, 1] = -s
        rot[:, 1, 0] = s
        rot[:, 1, 1] = c
        rot[:, 2, 2] = 1.0
        return rot
    raise ValueError(f"Unsupported rotation axis: {axis}")


def forward_kinematics(bvh: BvhData) -> np.ndarray:
    num_frames = bvh.num_frames
    num_joints = len(bvh.joints)
    positions = np.zeros((num_frames, num_joints, 3), dtype=np.float64)
    rotations = np.tile(np.eye(3, dtype=np.float64), (num_frames, num_joints, 1, 1))
    cursor = 0

    for joint_idx, joint in enumerate(bvh.joints):
        local_translation = np.repeat(joint.offset[None, :], num_frames, axis=0)
        local_rotation = np.tile(np.eye(3, dtype=np.float64), (num_frames, 1, 1))
        joint_channels = bvh.motion[:, cursor : cursor + len(joint.channels)]
        cursor += len(joint.channels)

        for channel_idx, channel_name in enumerate(joint.channels):
            axis_name = channel_name[0].upper()
            channel_values = joint_channels[:, channel_idx]
            if channel_name.endswith("position"):
                local_translation[:, AXIS_INDEX[axis_name]] += channel_values
            elif channel_name.endswith("rotation"):
                local_rotation = local_rotation @ rotation_matrix_batch(
                    axis_name, np.deg2rad(channel_values)
                )
            else:
                raise ValueError(f"Unsupported channel: {channel_name}")

        if joint.parent < 0:
            rotations[:, joint_idx] = local_rotation
            positions[:, joint_idx] = local_translation
            continue

        parent_rotation = rotations[:, joint.parent]
        parent_position = positions[:, joint.parent]
        rotations[:, joint_idx] = parent_rotation @ local_rotation
        positions[:, joint_idx] = parent_position + np.einsum(
            "fij,fj->fi", parent_rotation, local_translation
        )

    return positions


def zero_pose_positions(joints: Sequence[Joint]) -> np.ndarray:
    positions = np.zeros((len(joints), 3), dtype=np.float64)
    for joint_idx, joint in enumerate(joints):
        if joint.parent < 0:
            positions[joint_idx] = joint.offset
        else:
            positions[joint_idx] = positions[joint.parent] + joint.offset
    return positions


def infer_source_up_axis(joints: Sequence[Joint]) -> Tuple[str, str]:
    if joints and joints[0].children:
        first_child_offset = np.abs(joints[joints[0].children[0]].offset)
        y_mag = float(first_child_offset[1])
        z_mag = float(first_child_offset[2])
        if y_mag > 1e-6 or z_mag > 1e-6:
            axis = "y" if y_mag >= z_mag else "z"
            reason = f"root first-child offset |Y|={y_mag:.3f}, |Z|={z_mag:.3f}"
            return axis, reason

    rest_positions = zero_pose_positions(joints)
    y_span = float(rest_positions[:, 1].max() - rest_positions[:, 1].min())
    z_span = float(rest_positions[:, 2].max() - rest_positions[:, 2].min())
    axis = "y" if y_span >= z_span else "z"
    reason = f"rest-pose span Y={y_span:.3f}, Z={z_span:.3f}"
    return axis, reason


def normalize_source_up_axis(source_up: str, joints: Sequence[Joint]) -> Tuple[str, str]:
    if source_up == "auto":
        inferred, reason = infer_source_up_axis(joints)
        return inferred, f"auto ({reason})"
    return source_up, "manual override"


def reorient_positions(
    positions: np.ndarray, source_up: str, display_up: str
) -> Tuple[np.ndarray, str]:
    if display_up == "source":
        mapping = "display keeps source axes unchanged"
        return positions.copy(), mapping

    if display_up != "z":
        raise ValueError(f"Unsupported display_up mode: {display_up}")

    if source_up == "z":
        return positions.copy(), "display axes = source axes"

    if source_up == "y":
        remapped = positions.copy()
        remapped[..., 0] = positions[..., 0]
        remapped[..., 1] = -positions[..., 2]
        remapped[..., 2] = positions[..., 1]
        return remapped, "X_display = X_source, Y_display = -Z_source, Z_display = Y_source"

    raise ValueError(f"Unsupported source_up axis: {source_up}")


def build_edges(joints: Sequence[Joint]) -> List[Tuple[int, int]]:
    return [(joint.parent, joint_idx) for joint_idx, joint in enumerate(joints) if joint.parent >= 0]


def compute_axis_limits(positions: np.ndarray, margin_ratio: float = 0.08) -> Tuple[np.ndarray, float]:
    mins = positions.min(axis=(0, 1))
    maxs = positions.max(axis=(0, 1))
    center = (mins + maxs) / 2.0
    radius = float(np.max(maxs - mins) / 2.0)
    radius = max(radius * (1.0 + margin_ratio), 1.0)
    return center, radius


def set_equal_axes(ax, center: np.ndarray, radius: float) -> None:
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    if hasattr(ax, "set_box_aspect"):
        ax.set_box_aspect((1.0, 1.0, 1.0))


def add_display_axes(ax, anchor: np.ndarray, radius: float) -> None:
    axis_len = radius * 0.22
    ax.quiver(*anchor, axis_len, 0, 0, color="#d94841", linewidth=2)
    ax.quiver(*anchor, 0, axis_len, 0, color="#2b8a3e", linewidth=2)
    ax.quiver(*anchor, 0, 0, axis_len, color="#1971c2", linewidth=2)
    ax.text(*(anchor + np.array([axis_len * 1.08, 0, 0])), "X", color="#d94841")
    ax.text(*(anchor + np.array([0, axis_len * 1.08, 0])), "Y", color="#2b8a3e")
    ax.text(*(anchor + np.array([0, 0, axis_len * 1.08])), "Z", color="#1971c2")


def add_ground_plane(
    ax, positions: np.ndarray, center: np.ndarray, radius: float, display_up_axis: str
) -> None:
    plane_half = radius * 0.95
    if display_up_axis == "z":
        ground = float(positions[..., 2].min())
        verts = [
            [
                [center[0] - plane_half, center[1] - plane_half, ground],
                [center[0] + plane_half, center[1] - plane_half, ground],
                [center[0] + plane_half, center[1] + plane_half, ground],
                [center[0] - plane_half, center[1] + plane_half, ground],
            ]
        ]
    elif display_up_axis == "y":
        ground = float(positions[..., 1].min())
        verts = [
            [
                [center[0] - plane_half, ground, center[2] - plane_half],
                [center[0] + plane_half, ground, center[2] - plane_half],
                [center[0] + plane_half, ground, center[2] + plane_half],
                [center[0] - plane_half, ground, center[2] + plane_half],
            ]
        ]
    else:
        raise ValueError(f"Unsupported display up axis: {display_up_axis}")
    plane = Poly3DCollection(verts, facecolor="#adb5bd", edgecolor="none", alpha=0.12)
    ax.add_collection3d(plane)


def frame_subset(
    positions: np.ndarray, start_frame: int, end_frame: int, stride: int
) -> Tuple[np.ndarray, np.ndarray]:
    frame_ids = np.arange(start_frame, end_frame + 1, stride, dtype=np.int32)
    return positions[frame_ids], frame_ids


def add_overlay_text(fig, info_lines: Sequence[str]) -> None:
    fig.text(
        0.02,
        0.02,
        "\n".join(info_lines),
        fontsize=10,
        family="monospace",
        va="bottom",
        ha="left",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85, "edgecolor": "#ced4da"},
    )


def render_single_frame(
    positions: np.ndarray,
    frame_label: str,
    edges: Sequence[Tuple[int, int]],
    title_prefix: str,
    info_lines: Sequence[str],
    display_up_axis: str,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    center, radius = compute_axis_limits(positions[None, ...])
    add_ground_plane(ax, positions[None, ...], center, radius, display_up_axis)
    add_display_axes(ax, center - np.array([radius * 0.8, radius * 0.8, radius * 0.8]), radius)

    frame = positions
    ax.scatter(frame[:, 0], frame[:, 1], frame[:, 2], s=18, c="#1f77b4", depthshade=False)
    for parent_idx, joint_idx in edges:
        segment = frame[[parent_idx, joint_idx]]
        ax.plot(segment[:, 0], segment[:, 1], segment[:, 2], color="#212529", linewidth=1.8)

    ax.set_title(f"{title_prefix}\n{frame_label}")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    set_equal_axes(ax, center, radius)
    add_overlay_text(fig, info_lines)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def animate_skeleton(
    positions: np.ndarray,
    frame_ids: np.ndarray,
    playback_fps: float,
    edges: Sequence[Tuple[int, int]],
    title_prefix: str,
    info_lines: Sequence[str],
    show_root_traj: bool,
    display_up_axis: str,
    total_frames: int,
) -> None:
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    center, radius = compute_axis_limits(positions)
    add_ground_plane(ax, positions, center, radius, display_up_axis)
    add_display_axes(ax, center - np.array([radius * 0.8, radius * 0.8, radius * 0.8]), radius)
    set_equal_axes(ax, center, radius)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    add_overlay_text(
        fig,
        list(info_lines)
        + [
            "controls: space pause/resume, left/right single-step, q close",
        ],
    )

    initial = positions[0]
    points = ax.scatter(initial[:, 0], initial[:, 1], initial[:, 2], s=18, c="#1f77b4", depthshade=False)
    lines = [
        ax.plot(
            [initial[parent_idx, 0], initial[joint_idx, 0]],
            [initial[parent_idx, 1], initial[joint_idx, 1]],
            [initial[parent_idx, 2], initial[joint_idx, 2]],
            color="#212529",
            linewidth=1.8,
        )[0]
        for parent_idx, joint_idx in edges
    ]
    root_traj_line = None
    if show_root_traj:
        root = positions[:, 0]
        root_traj_line = ax.plot(
            root[:, 0], root[:, 1], root[:, 2], color="#e8590c", linewidth=1.1, alpha=0.75
        )[0]

    state = {"paused": False, "frame_cursor": 0}

    def draw_frame(local_idx: int) -> None:
        frame = positions[local_idx]
        points._offsets3d = (frame[:, 0], frame[:, 1], frame[:, 2])
        for line_artist, (parent_idx, joint_idx) in zip(lines, edges):
            segment = frame[[parent_idx, joint_idx]]
            line_artist.set_data(segment[:, 0], segment[:, 1])
            line_artist.set_3d_properties(segment[:, 2])
        if root_traj_line is not None:
            history = positions[: local_idx + 1, 0]
            root_traj_line.set_data(history[:, 0], history[:, 1])
            root_traj_line.set_3d_properties(history[:, 2])
        ax.set_title(f"{title_prefix}\nframe {int(frame_ids[local_idx]) + 1}/{total_frames}")

    def update(local_idx: int):
        state["frame_cursor"] = local_idx
        draw_frame(local_idx)
        return lines + [points]

    anim = FuncAnimation(
        fig,
        update,
        frames=len(frame_ids),
        interval=1000.0 / playback_fps,
        blit=False,
        repeat=True,
    )

    def on_key(event) -> None:
        if event.key == " ":
            state["paused"] = not state["paused"]
            if state["paused"]:
                anim.event_source.stop()
            else:
                anim.event_source.start()
        elif event.key == "right":
            state["paused"] = True
            anim.event_source.stop()
            state["frame_cursor"] = (state["frame_cursor"] + 1) % len(frame_ids)
            draw_frame(state["frame_cursor"])
            fig.canvas.draw_idle()
        elif event.key == "left":
            state["paused"] = True
            anim.event_source.stop()
            state["frame_cursor"] = (state["frame_cursor"] - 1) % len(frame_ids)
            draw_frame(state["frame_cursor"])
            fig.canvas.draw_idle()
        elif event.key == "q":
            plt.close(fig)

    fig.canvas.mpl_connect("key_press_event", on_key)
    draw_frame(0)
    plt.tight_layout()
    plt.show()


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="用点线火柴人可视化 BVH，并明确标出源坐标系是 Y-up 还是 Z-up。"
    )
    parser.add_argument("bvh_path", type=Path, help="输入 BVH 文件路径")
    parser.add_argument(
        "--source-up",
        choices=["auto", "y", "z"],
        default="auto",
        help="BVH 源坐标系的 up 轴。auto 会根据骨架偏移做启发式判断。",
    )
    parser.add_argument(
        "--display-up",
        choices=["z", "source"],
        default="z",
        help="显示坐标系的 up 轴。默认旋到 Z-up 方便看；source 保持 BVH 原始坐标。",
    )
    parser.add_argument("--start-frame", type=int, default=0, help="起始帧，0-based")
    parser.add_argument("--end-frame", type=int, default=None, help="结束帧，0-based，默认到最后一帧")
    parser.add_argument("--stride", type=int, default=1, help="抽帧步长")
    parser.add_argument("--fps", type=float, default=None, help="播放 FPS，默认使用 BVH 原始帧率")
    parser.add_argument(
        "--save-preview",
        type=Path,
        default=None,
        help="只渲染一帧并保存图片，不弹交互窗口",
    )
    parser.add_argument(
        "--preview-frame",
        type=int,
        default=None,
        help="配合 --save-preview 使用，指定保存哪一帧，默认使用 start-frame",
    )
    parser.add_argument(
        "--hide-root-trajectory",
        action="store_true",
        help="隐藏 root 历史轨迹",
    )
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    bvh = parse_bvh(args.bvh_path)
    raw_positions = forward_kinematics(bvh)
    source_up, source_reason = normalize_source_up_axis(args.source_up, bvh.joints)
    display_positions, mapping_text = reorient_positions(raw_positions, source_up, args.display_up)
    edges = build_edges(bvh.joints)

    end_frame = bvh.num_frames - 1 if args.end_frame is None else args.end_frame
    if not (0 <= args.start_frame < bvh.num_frames):
        raise ValueError(f"start-frame out of range: {args.start_frame}")
    if not (args.start_frame <= end_frame < bvh.num_frames):
        raise ValueError(f"end-frame out of range: {end_frame}")
    if args.stride <= 0:
        raise ValueError(f"stride must be > 0, got {args.stride}")

    subset_positions, subset_frame_ids = frame_subset(
        display_positions, args.start_frame, end_frame, args.stride
    )
    playback_fps = args.fps if args.fps is not None else 1.0 / bvh.frame_time
    fps_text = f"{playback_fps:.3f}"
    source_up_text = source_up.upper()
    display_up_text = "source axes" if args.display_up == "source" else "Z-up"
    display_up_axis = source_up if args.display_up == "source" else "z"
    ground_plane_text = "XZ in display space" if display_up_axis == "y" else "XY in display space"

    print(f"BVH file: {args.bvh_path}")
    print(
        f"frames: {bvh.num_frames}, frame_time: {bvh.frame_time:.6f}s, source_fps: {1.0 / bvh.frame_time:.3f}"
    )
    print(f"joints: {len(bvh.joints)}, edges: {len(edges)}, channels: {bvh.num_channels}")
    print(f"source up-axis: {source_up_text}-up ({source_reason})")
    print(f"display orientation: {display_up_text}")
    print(f"axis mapping: {mapping_text}")

    title_prefix = (
        f"{args.bvh_path.name} | source: {source_up_text}-up | display: {display_up_text} | fps: {fps_text}"
    )
    info_lines = [
        f"source up-axis : {source_up_text}-up",
        f"source decision: {source_reason}",
        f"display mode   : {display_up_text}",
        f"axis mapping   : {mapping_text}",
        f"ground plane   : {ground_plane_text}",
    ]

    if args.save_preview is not None:
        preview_frame = args.start_frame if args.preview_frame is None else args.preview_frame
        if not (args.start_frame <= preview_frame <= end_frame):
            raise ValueError(
                f"preview-frame must be within [{args.start_frame}, {end_frame}], got {preview_frame}"
            )
        preview_local = (preview_frame - args.start_frame) // args.stride
        if subset_frame_ids[preview_local] != preview_frame:
            raise ValueError(
                f"preview-frame {preview_frame} does not align with stride={args.stride} from start-frame={args.start_frame}"
            )
        render_single_frame(
            subset_positions[preview_local],
            frame_label=f"frame {preview_frame + 1}/{bvh.num_frames}",
            edges=edges,
            title_prefix=title_prefix,
            info_lines=info_lines,
            display_up_axis=display_up_axis,
            output_path=args.save_preview,
        )
        print(f"saved preview: {args.save_preview}")
        return

    animate_skeleton(
        positions=subset_positions,
        frame_ids=subset_frame_ids,
        playback_fps=playback_fps,
        edges=edges,
        title_prefix=title_prefix,
        info_lines=info_lines,
        show_root_traj=not args.hide_root_trajectory,
        display_up_axis=display_up_axis,
        total_frames=bvh.num_frames,
    )


if __name__ == "__main__":
    main()
