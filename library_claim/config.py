"""Load and validate shared experiment settings from the project YAML file."""
import hashlib
from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, PositiveFloat, PositiveInt, model_validator
import yaml

ROOT = Path(__file__).resolve().parent.parent
Unit = Annotated[float, Field(ge=0, le=1)]
Byte = Annotated[int, Field(ge=0, le=255)]
Text = Annotated[str, Field(min_length=1)]


class Settings(BaseModel):
    """Reject misspelled settings and nonfinite numeric experiments."""
    model_config = ConfigDict(extra='forbid', frozen=True, allow_inf_nan=False)


class Artifact(Settings):
    """Identify one pinned downloadable model artifact."""
    remote: Text
    sha256: Annotated[str, Field(pattern=r'^[a-f0-9]{64}$')]


class Prompts(Settings):
    """Define the two supported semantic targets without a catch-all class."""
    books: Annotated[list[Text], Field(min_length=1)]
    empty_background: Annotated[list[Text], Field(min_length=1)]


class Artifacts(Settings):
    """Name the required compatible CLIP encoder and tokenizer artifacts."""
    vision: Artifact
    text: Artifact
    tokenizer: Artifact


class RelevanceConfig(Settings):
    """Configure local classification, preprocessing, and model setup."""
    model_id: Text
    revision: Text
    model_dir: Path
    method: Text
    min_similarity: Annotated[float, Field(ge=-1, le=1)]
    min_margin: Annotated[float, Field(ge=0, le=2)]
    prompts: Prompts
    image_size: PositiveInt
    image_mean: tuple[Unit, Unit, Unit]
    image_std: tuple[PositiveFloat, PositiveFloat, PositiveFloat]
    padding_rgb: tuple[Byte, Byte, Byte]
    token_length: PositiveInt
    pad_id: Annotated[int, Field(ge=0)]
    pad_token: Text
    intra_op_threads: PositiveInt
    inter_op_threads: PositiveInt
    download_timeout_s: PositiveFloat
    artifacts: Artifacts


class QualityConfig(Settings):
    """Configure measurement and conservative relevance admission thresholds."""
    max_dimension: Annotated[int, Field(ge=32)]
    dark_cutoff: Byte
    bright_cutoff: Byte
    brisque_model: Path
    brisque_range: Path
    method: Text
    brisque_ranking_scale: PositiveFloat
    max_extreme_fraction: Unit
    min_sharpness: Annotated[float, Field(ge=0)]

    @model_validator(mode='after')
    def ordered_cutoffs(self):
        """Reject overlapping dark and bright ranges.

        Returns:
            The validated quality settings.

        Raises:
            ValueError: Dark cutoff is not below bright cutoff.
        """
        if self.dark_cutoff >= self.bright_cutoff:
            raise ValueError('dark_cutoff must be below bright_cutoff')
        return self


class TrackingConfig(Settings):
    """Configure bounded feature extraction, retrieval, flow, and geometry."""
    max_dimension: Annotated[int, Field(ge=32)]
    capacity: PositiveInt
    reanchor_every: PositiveInt
    max_gap_s: PositiveFloat
    min_reference_overlap: Unit
    sift_features: PositiveInt
    flann_trees: PositiveInt
    flann_checks: PositiveInt
    descriptor_ratio: Annotated[float, Field(gt=0, lt=1)]
    candidate_limit: PositiveInt
    min_matches: Annotated[int, Field(ge=4)]
    reprojection_threshold_px: PositiveFloat
    ransac_iterations: PositiveInt
    ransac_confidence: Annotated[float, Field(gt=0, lt=1)]
    min_inlier_fraction: Unit
    min_support_area: Unit
    min_projected_area_ratio: PositiveFloat
    max_projected_area_ratio: PositiveFloat
    forward_backward_error_px: PositiveFloat
    max_corners: PositiveInt
    corner_quality: Annotated[float, Field(gt=0, le=1)]
    corner_min_distance_px: PositiveFloat
    lk_window: PositiveInt
    lk_max_level: Annotated[int, Field(ge=0)]
    lk_iterations: PositiveInt
    lk_epsilon: PositiveFloat
    lk_min_eigen_threshold: PositiveFloat

    @model_validator(mode='after')
    def ordered_areas(self):
        """Validate projected-area bounds before geometry runs.

        Returns:
            The validated tracking settings.

        Raises:
            ValueError: The lower area bound is not smaller than the upper bound.
        """
        if self.min_projected_area_ratio >= self.max_projected_area_ratio:
            raise ValueError('Projected area bounds must be increasing')
        return self


class SelectionConfig(Settings):
    """Configure evidence improvement heuristics."""
    new_coverage_fraction: Unit
    detail_gain: Annotated[float, Field(gt=1)]
    brisque_improvement: PositiveFloat
    sharpness_gain: Annotated[float, Field(gt=1)]
    sharpness_floor: Annotated[float, Field(ge=0)]


class BrowserConfig(Settings):
    """Expose camera capture settings that contain no server credentials."""
    frame_interval_ms: PositiveInt
    frame_width: PositiveInt
    ideal_width: PositiveInt
    ideal_height: PositiveInt
    jpeg_quality: Annotated[float, Field(gt=0, le=1)]


class CaptureConfig(Settings):
    """Configure live processing and logging cadences."""
    quality_interval_s: PositiveFloat
    summary_log_interval_s: PositiveFloat
    remote_interval_s: PositiveFloat
    frame_max_age_s: PositiveFloat
    browser: BrowserConfig


class SchedulerConfig(Settings):
    """Configure default job budgets."""
    concurrency: PositiveInt
    pending_limit: PositiveInt
    timeout_s: PositiveFloat


class InferenceConfig(Settings):
    """Configure shared remote model and generation defaults."""
    model: Text
    vision_model: Text
    max_tokens: PositiveInt
    conversation_max_tokens: PositiveInt
    temperature: Annotated[float, Field(ge=0, le=2)]
    qwen_enable_thinking: bool
    timeout_s: PositiveFloat
    connect_timeout_s: PositiveFloat


class SpeechConfig(Settings):
    """Configure speech models and transport budgets."""
    stt_model: Text
    tts_model: Text
    tts_chunk_chars: PositiveInt
    timeout_s: PositiveFloat
    connect_timeout_s: PositiveFloat
    close_timeout_s: PositiveFloat


class ExperimentConfig(Settings):
    """Configure offline comparisons independently of live classifier behavior."""
    edge_crop_count: Annotated[int, Field(ge=2)]


class AppConfig(Settings):
    """Collect the independently owned experiment setting groups."""
    experiments: ExperimentConfig
    relevance: RelevanceConfig
    quality: QualityConfig
    tracking: TrackingConfig
    selection: SelectionConfig
    capture: CaptureConfig
    scheduler: SchedulerConfig
    inference: InferenceConfig
    speech: SpeechConfig


def load_config(path: Path = ROOT / 'config.yml') -> AppConfig:
    """Read safe YAML and reject missing, unknown, or invalid settings.

    Args:
        path: Configuration document to validate.

    Returns:
        Validated settings, independent of the caller's working directory.

    Raises:
        OSError: The configuration file cannot be read.
        yaml.YAMLError: The document is invalid YAML.
        pydantic.ValidationError: A setting violates the configuration contract.
    """
    return AppConfig.model_validate(yaml.safe_load(path.read_text()))


CONFIG = load_config()
# Keep the settings used for an observation identifiable after later experiments.
CONFIG_DIGEST = hashlib.sha256(CONFIG.model_dump_json().encode()).hexdigest()
