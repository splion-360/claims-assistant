"""Define validated records for sweeps, books, observations, and processing jobs."""
from typing import Annotated, Any, Literal
from uuid import uuid4

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator

Finite = Annotated[float, Field(allow_inf_nan=False)]
Score = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
Centimetres = Annotated[float, Field(gt=0, allow_inf_nan=False)]
Identifier = Annotated[str, Field(pattern=r'^[A-Za-z0-9_-]{1,128}$')]


def new_id() -> str:
    """Generate a unique record identifier.

    Returns:
        A UUID represented as a hexadecimal string.
    """
    return uuid4().hex


class Record(BaseModel):
    """Define a frozen record with a stable identifier."""
    model_config = ConfigDict(extra='forbid', frozen=True)
    id: Identifier = Field(default_factory=new_id)


class Sweep(Record):
    """Record the capture session and its current sweep state."""
    captured_at: AwareDatetime
    device: str = ''
    country: str = ''
    currency: str = ''
    state: Literal['Sweeping', 'Approaching', 'Capturing', 'Resuming'] = 'Sweeping'
    active_target: Identifier | None = None
    context_revision: Annotated[int, Field(ge=0)] = 0
    # The camera stopping starts the time-to-packet clock; background work finishing stops it.
    ended_at: AwareDatetime | None = None
    packet_ready_at: AwareDatetime | None = None


class Box(BaseModel):
    """Normalized image coordinates; independent of image resolution."""
    model_config = ConfigDict(extra='forbid', frozen=True)
    x_min: Score
    y_min: Score
    x_max: Score
    y_max: Score

    @model_validator(mode='after')
    def ordered(self):
        """Validate that the box encloses a positive image area.

        Returns:
            The validated box.

        Raises:
            ValueError: The coordinate bounds have zero or negative extent.
        """
        if self.x_min >= self.x_max or self.y_min >= self.y_max:
            raise ValueError('Box must have positive width and height')
        return self


class Book(Record):
    """Store the accepted identity of a physical book copy."""
    sweep_id: Identifier
    shelf_id: Identifier | None = None
    position: Annotated[int, Field(ge=0)] | None = None
    excluded: bool = False
    association_status: Literal['provisional', 'confirmed', 'uncertain'] = 'provisional'
    identification_status: Literal['identified', 'unidentified'] = 'unidentified'
    needs_appraisal: bool = False
    title: str | None = None
    author: str | None = None
    publisher: str | None = None
    edition: str | None = None
    isbn: str | None = None
    id_confidence: Score | None = None


class Quality(BaseModel):
    """Optional signals, not calibrated probabilities; None means unmeasured."""
    model_config = ConfigDict(extra='forbid', frozen=True)
    sharpness: Annotated[float, Field(ge=0, allow_inf_nan=False)] | None = None
    brisque: Annotated[float, Field(allow_inf_nan=False)] | None = None
    # Bright pixels are measurable; glare requires contextual interpretation.
    bright_fraction: Score | None = None
    temporal_change: Score | None = None
    glare_fraction: Score | None = None
    dark_fraction: Score | None = None


class Frame(Record):
    """Reference a saved image and its capture metadata."""
    sweep_id: Identifier
    captured_at: AwareDatetime
    evidence_key: Annotated[str, Field(pattern=r'^[a-f0-9]{64}\.(jpg|png)$')]
    width_px: Annotated[int, Field(gt=0)]
    height_px: Annotated[int, Field(gt=0)]
    selection_reason: str | None = None
    quality: Quality | None = None
    quality_metadata: dict[str, Any] | None = None
    relevance_metadata: dict[str, Any] | None = None



class BookObservation(Record):
    """Associate frame evidence and extracted facts with a physical book."""
    sweep_id: Identifier
    book_id: Identifier
    frame_id: Identifier
    bbox: Box
    polygon: Annotated[list[tuple[Score, Score]], Field(min_length=4, max_length=4)] | None = None
    view: Literal['spine', 'cover', 'pages', 'partial'] | None = None
    visible_text: str | None = None
    visual_issues: list[str] = Field(default_factory=list)
    quality_metadata: dict[str, Any] | None = None
    association_method: str | None = None
    association_confidence: Score | None = None
    quality: Quality = Field(default_factory=Quality)
    title: str | None = None
    author: str | None = None
    publisher: str | None = None
    edition: str | None = None
    isbn: str | None = None
    provenance: Literal['visual', 'user_statement', 'catalogue'] = 'visual'
    source_model: str | None = None
    unresolved_fields: list[str] = Field(default_factory=list)
    # Metric size from a depth map; empty when no depth was available for the frame.
    spine_height_cm: Centimetres | None = None
    spine_thickness_cm: Centimetres | None = None
    scale_method: Literal['webxr_depth', 'depth_pro'] | None = None
    # Why this sighting produced no size, when it produced none.
    size_reason: str | None = None
    # Where a catalogue observation came from and when it was fetched.
    catalogue_url: str | None = None
    retrieved_at: AwareDatetime | None = None


class Item(Record):
    """Store the accepted identity of one physical non-book object."""
    sweep_id: Identifier
    excluded: bool = False
    association_status: Literal['provisional', 'confirmed', 'uncertain'] = 'provisional'
    category: str = 'other'
    description: str = ''
    material: str | None = None
    brand_model: str | None = None
    # Art and mirrors wait for a person unless the policyholder says the piece is a print.
    needs_appraisal: bool = False
    stated_print: bool = False
    # True once a person has settled this object's review.
    reviewed: bool = False


class ItemObservation(Record):
    """Associate frame evidence and extracted facts with a physical object."""
    sweep_id: Identifier
    item_id: Identifier
    frame_id: Identifier
    bbox: Box
    category: str
    description: str
    material: str | None = None
    brand_model: str | None = None
    visual_issues: list[str] = Field(default_factory=list)
    # Item review reasons live on the observation: review_issues keys on books.
    review_reasons: list[str] = Field(default_factory=list)
    association_method: str | None = None
    quality: Quality = Field(default_factory=Quality)
    source_model: str | None = None
    width_cm: Centimetres | None = None
    height_cm: Centimetres | None = None
    depth_cm: Centimetres | None = None
    scale_method: Literal['webxr_depth', 'depth_pro'] | None = None


class ReviewIssue(Record):
    """Track an unresolved field requiring review."""
    sweep_id: Identifier
    book_id: Identifier
    field: str
    reason: Annotated[str, Field(min_length=1)]
    status: Literal['open', 'resolved'] = 'open'


class ProcessingJob(Record):
    """Persist a processing request and its execution status."""
    sweep_id: Identifier
    frame_id: Identifier
    book_id: Identifier | None = None
    task: Literal['identify', 'items', 'measure', 'room', 'value', 'value_item', 'visual_diagnosis', 'frame_check']
    status: Literal['pending', 'running', 'succeeded', 'failed', 'cancelled'] = 'pending'
    created_at: AwareDatetime
    model: str | None = None
    configuration_version: str = '1'
    target_id: Identifier | None = None
    evidence_score: Score = 0
    context_revision: Annotated[int, Field(ge=0)] = 0
    result: dict[str, Any] | None = None
    error: str | None = None


class FrameAssociation(Record):
    """Persist a verified normalized current-image to reference-image mapping."""

    sweep_id: Identifier
    frame_id: Identifier
    reference_frame_id: Identifier
    matrix: tuple[tuple[Finite, Finite, Finite], tuple[Finite, Finite, Finite], tuple[Finite, Finite, Finite]]
    method: Literal['optical_flow', 'sift_ransac']
    # Four correspondences are the planar minimum; config controls admission above it.
    matched_points: Annotated[int, Field(ge=4)]
    reprojection_error_px: Annotated[float, Field(ge=0, allow_inf_nan=False)]

    @model_validator(mode='after')
    def invertible(self):
        """Require a nonsingular projective mapping.

        Returns:
            The validated association.

        Raises:
            ValueError: The matrix is singular.
        """
        a, b, c = self.matrix
        determinant = (a[0] * (b[1] * c[2] - b[2] * c[1])
                       - a[1] * (b[0] * c[2] - b[2] * c[0])
                       + a[2] * (b[0] * c[1] - b[1] * c[0]))
        if abs(determinant) < 1e-10:
            raise ValueError('Mapping must be invertible')
        return self
