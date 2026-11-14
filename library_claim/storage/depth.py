"""Store per-frame metric depth as sidecar files beside the saved JPEG evidence."""
import base64
import binascii
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


# Upper bound on browser depth grids so one frame cannot allocate unbounded memory.
MAX_GRID = (160, 120)


def parse_depth_payload(payload: dict, image_width: int, image_height: int) -> DepthMap:
    """Decode a WebXR depth grid sent alongside a camera JPEG.

    Args:
        payload: `width`, `height`, base64 little-endian uint16 `mm`, and `fx`, `fy`, `cx`, `cy`
            already expressed in JPEG pixels.
        image_width: Decoded JPEG width.
        image_height: Decoded JPEG height.

    Returns:
        A WebXR depth map in metres; zero cells mean unknown depth.

    Raises:
        ValueError: The grid size, byte count or intrinsics are invalid.
    """
    try:
        width, height = int(payload['width']), int(payload['height'])
        raw = base64.b64decode(payload['mm'], validate=True)
        fx, fy, cx, cy = (float(payload[key]) for key in ('fx', 'fy', 'cx', 'cy'))
    except (KeyError, TypeError, ValueError, binascii.Error):
        raise ValueError('Malformed depth payload') from None
    if not (0 < width <= MAX_GRID[0] and 0 < height <= MAX_GRID[1]):
        raise ValueError('Depth grid size is out of range')
    if len(raw) != width * height * 2:
        raise ValueError('Depth grid byte count does not match its size')
    if not all(np.isfinite(value) for value in (fx, fy, cx, cy)) or fx <= 0 or fy <= 0:
        raise ValueError('Depth intrinsics are invalid')
    grid = np.frombuffer(raw, '<u2').reshape(height, width).astype(np.float32) / 1000
    return DepthMap(depth_m=grid, fx=fx, fy=fy, cx=cx, cy=cy, image_width=image_width,
                    image_height=image_height, method='webxr_depth')
