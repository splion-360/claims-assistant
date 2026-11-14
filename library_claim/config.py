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


class IdentificationConfig(Settings):
    """Bound visual extraction and conservative physical-copy association."""
    guidance_cooldown_s: PositiveFloat
    guidance_timeout_s: PositiveFloat
    guidance_max_requests: PositiveInt
    recapture_max_attempts: PositiveInt
    max_books_per_frame: PositiveInt
    association_min_iou: Unit
    association_max_hops: PositiveInt
    text_min_characters: PositiveInt
    text_edit_similarity: Unit
    visual_min_crop_size: PositiveInt
    visual_max_dimension: PositiveInt
    visual_max_features: PositiveInt
    visual_ratio: Unit
    visual_min_matches: PositiveInt
    visual_min_inlier_fraction: Unit
    visual_min_support: Unit
    visual_reprojection_px: PositiveFloat
    visual_ransac_iterations: PositiveInt
    visual_ransac_confidence: Unit


class QualityConfig(Settings):
    """Configure frame measurement and the local exposure/sharpness gate."""
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
    depth_grid_width: Annotated[int, Field(gt=0, le=160)]
    depth_grid_height: Annotated[int, Field(gt=0, le=120)]


class CaptureConfig(Settings):
    """Configure live processing and logging cadences."""
    quality_interval_s: PositiveFloat
    summary_log_interval_s: PositiveFloat
    remote_interval_s: PositiveFloat
    frame_max_age_s: PositiveFloat
    max_reads_per_session: PositiveInt
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
    fallback_models: list[Text]
    vision_fallback_models: list[Text]
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


class DrexConfig(Settings):
    """Configure the Drex decision endpoint and decision thresholds."""
    model: Text
    base_url: Text
    client_name: Text
    timeout_s: PositiveFloat
    connect_timeout_s: PositiveFloat
    book_visible_min: Unit
    legible_min: Unit
    capture_defects: list[Text]
    same_book_min: Unit
    different_book_max: Unit
    catalogue_min: Unit
    target_min: Unit
    quality_prompt_cooldown_s: PositiveFloat
    same_prompt_cooldown_s: PositiveFloat
    gate_failure_streak: PositiveInt
    quiet_streak: PositiveInt
    quiet_interval_s: PositiveFloat

    @model_validator(mode='after')
    def ordered_same_book_bounds(self):
        """Keep an unresolved band between the different and same thresholds.

        Returns:
            The validated settings.

        Raises:
            ValueError: The different-book bound is not below the same-book bound.
        """
        if self.different_book_max >= self.same_book_min:
            raise ValueError('different_book_max must be below same_book_min')
        return self


class MeasurementConfig(Settings):
    """Configure metric depth sampling and the fallback depth model."""
    enabled: bool
    depth_inset: Annotated[float, Field(ge=0, lt=0.5)]
    depth_window_px: PositiveInt
    depth_tolerance: Unit
    height_range_cm: tuple[PositiveFloat, PositiveFloat]
    thickness_range_cm: tuple[PositiveFloat, PositiveFloat]
    depth_pro_model: Text
    # Room geometry: depth readings outside this range are ignored, and an estimate
    # needs at least this many posed frames and points before it is reported.
    room_depth_range_m: tuple[PositiveFloat, PositiveFloat]
    room_min_frames: PositiveInt
    room_min_points: PositiveInt
    room_yaw_bins: PositiveInt
    # Below this share of a full turn, the floor outline is flagged as partial.
    room_min_coverage: Unit


class MarketConfig(Settings):
    """One country's currency and the price providers tried for it, in order."""
    currency: Text
    providers: list[Text]


class PricingConfig(Settings):
    """Pricing locale, provider behaviour and the bar for human appraisal."""
    default_country: Text
    condition_assumed: Text
    # A book worth more than this in the sweep currency is withheld for appraisal.
    appraisal_threshold: PositiveFloat
    appraisal_keywords: list[Text]
    convert_from: list[Text]
    # Retail providers for non-book contents; empty means those lines stay unpriced.
    item_providers: list[Text]
    fx_base_url: Text
    timeout_s: PositiveFloat
    connect_timeout_s: PositiveFloat
    markets: dict[Text, MarketConfig]

    @model_validator(mode='after')
    def default_country_is_configured(self):
        """Require a market for the default country.

        Returns:
            The validated settings.

        Raises:
            ValueError: The default country has no configured market.
        """
        if self.default_country and self.default_country.upper() not in self.markets:
            raise ValueError(f'No market configured for default_country {self.default_country}')
        return self


class ItemsConfig(Settings):
    """Non-book detection limits, association geometry and the appraisal rule."""
    enabled: bool
    max_items_per_frame: PositiveInt
    # Furniture does not move, so objects are inspected on a cadence, not every frame.
    detection_interval_s: PositiveFloat
    # Two detections of one object in a shared coordinate basis must overlap by this much.
    association_min_iou: Unit
    # Wording that lets a policyholder release a framed piece from appraisal.
    print_statements: list[Text]


class AppConfig(Settings):
    """Collect the independently owned setting groups."""
    identification: IdentificationConfig
    quality: QualityConfig
    tracking: TrackingConfig
    selection: SelectionConfig
    capture: CaptureConfig
    scheduler: SchedulerConfig
    inference: InferenceConfig
    speech: SpeechConfig
    drex: DrexConfig
    measurement: MeasurementConfig
    pricing: PricingConfig
    items: ItemsConfig


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
