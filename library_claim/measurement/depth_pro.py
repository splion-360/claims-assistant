"""Estimate metric depth and focal length for a frame with Apple Depth Pro."""
from functools import lru_cache
import logging
import time

import cv2
import numpy as np

from library_claim.config import CONFIG
from library_claim.measurement.spine import DepthMap

logger = logging.getLogger(__name__)


class DepthUnavailable(RuntimeError):
    """Depth could not be estimated; books are then saved without dimensions."""


# Set after a CUDA out-of-memory error so later frames go straight to the CPU.
_cpu_only = False


def _run(jpeg: bytes, device: str) -> tuple[np.ndarray, float, int, int]:
    """Run Depth Pro on one device.

    Args:
        jpeg: Encoded source frame.
        device: 'cuda' or 'cpu'.

    Returns:
        Depth in metres at image resolution, focal length in pixels, width and height.

    Raises:
        ValueError: The image cannot be decoded.
    """
    import torch

    processor, model, device = _load(device)
    image = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError('Cannot decode frame')
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width = rgb.shape[:2]
    inputs = processor(images=rgb, return_tensors='pt').to(device)
    inputs['pixel_values'] = inputs['pixel_values'].to(model.dtype)
    with torch.no_grad():
        outputs = model(**inputs)
    # Post-processing rescales depth to metres using the predicted field of view.
    result = processor.post_process_depth_estimation(outputs, target_sizes=[(height, width)])[0]
    return result['predicted_depth'].float().cpu().numpy(), float(result['focal_length']), width, height


@lru_cache(maxsize=2)
def _load(device: str):
    """Load the processor and model once per device, in fp16 on CUDA.

    Args:
        device: 'cuda' or 'cpu'.

    Returns:
        Processor, model and torch device.
    """
    import torch
    from transformers import DepthProForDepthEstimation, DepthProImageProcessor

    dtype = torch.float16 if device == 'cuda' else torch.float32
    processor = DepthProImageProcessor.from_pretrained(CONFIG.measurement.depth_pro_model)
    model = DepthProForDepthEstimation.from_pretrained(CONFIG.measurement.depth_pro_model, torch_dtype=dtype)
    return processor, model.to(device).eval(), device


def estimate_depth(jpeg: bytes) -> DepthMap:
    """Predict metric depth for one JPEG using the model's own focal-length estimate.

    Args:
        jpeg: Encoded source frame.

    Returns:
        Depth at full image resolution, focal length in image pixels, principal point at the centre.

    Raises:
        DepthUnavailable: The model could not be loaded or run, or the image is invalid.
    """
    global _cpu_only
    started = time.monotonic()
    try:
        import torch

        device = 'cuda' if torch.cuda.is_available() and not _cpu_only else 'cpu'
        try:
            depth, focal, width, height = _run(jpeg, device)
        except torch.cuda.OutOfMemoryError:
            # A small laptop GPU cannot hold Depth Pro; free it and stay on the CPU.
            _cpu_only, device = True, 'cpu'
            _load.cache_clear()
            torch.cuda.empty_cache()
            depth, focal, width, height = _run(jpeg, device)
    except Exception as error:
        raise DepthUnavailable(f'Depth Pro unavailable: {type(error).__name__}') from None
    logger.info('Depth Pro device=%s latency_ms=%.0f focal_px=%.0f', device, (time.monotonic() - started) * 1000, focal)
    return DepthMap(depth_m=depth.astype(np.float32), fx=focal, fy=focal, cx=width / 2, cy=height / 2,
                    image_width=width, image_height=height, method='depth_pro')
