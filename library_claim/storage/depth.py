"""Store per-frame metric depth as sidecar files beside the saved JPEG evidence."""
from pathlib import Path

import numpy as np

from library_claim.measurement.spine import DepthMap


def _path(root: Path, frame_id: str) -> Path:
    """Locate a frame's depth sidecar.

    Args:
        root: Capture data directory.
        frame_id: Saved frame identifier.

    Returns:
        Path of the `.npz` sidecar.
    """
    return Path(root) / 'depth' / f'{frame_id}.npz'


def save_depth(root: Path, frame_id: str, depth: DepthMap) -> None:
    """Write a depth map and its intrinsics for a saved frame.

    Args:
        root: Capture data directory.
        frame_id: Saved frame identifier.
        depth: Depth map for that exact frame.
    """
    path = _path(root, frame_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, depth_m=depth.depth_m.astype(np.float32),
                        intrinsics=np.array([depth.fx, depth.fy, depth.cx, depth.cy], np.float64),
                        size=np.array([depth.image_width, depth.image_height], np.int64),
                        method=np.array(depth.method))


def load_depth(root: Path, frame_id: str) -> DepthMap | None:
    """Read a frame's depth sidecar if one was captured.

    Args:
        root: Capture data directory.
        frame_id: Saved frame identifier.

    Returns:
        The depth map, or None when the frame has no sidecar.
    """
    path = _path(root, frame_id)
    if not path.exists():
        return None
    with np.load(path) as data:
        fx, fy, cx, cy = (float(value) for value in data['intrinsics'])
        width, height = (int(value) for value in data['size'])
        return DepthMap(depth_m=data['depth_m'], fx=fx, fy=fy, cx=cx, cy=cy,
                        image_width=width, image_height=height, method=str(data['method']))
