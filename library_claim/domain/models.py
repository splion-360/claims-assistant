"""Internal capture contracts; these are not the final claim-packet schema."""
from typing import Annotated, Any, Literal
from uuid import uuid4

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator

Score = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
Identifier = Annotated[str, Field(pattern=r'^[A-Za-z0-9_-]{1,128}$')]


def new_id() -> str:
    return uuid4().hex


class Record(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)
    id: Identifier = Field(default_factory=new_id)


class Sweep(Record):
    captured_at: AwareDatetime
    device: str = ''
    country: str = ''
    currency: str = ''
    state: Literal['Sweeping', 'Approaching', 'Capturing', 'Resuming'] = 'Sweeping'


class Frame(Record):
    sweep_id: Identifier
    captured_at: AwareDatetime
    evidence_key: Annotated[str, Field(pattern=r'^[a-f0-9]{64}\.(jpg|png)$')]
    width_px: Annotated[int, Field(gt=0)]
    height_px: Annotated[int, Field(gt=0)]


class Box(BaseModel):
    """Normalized image coordinates; independent of image resolution."""
    model_config = ConfigDict(extra='forbid', frozen=True)
    x_min: Score
    y_min: Score
    x_max: Score
    y_max: Score

    @model_validator(mode='after')
    def ordered(self):
        if self.x_min >= self.x_max or self.y_min >= self.y_max:
            raise ValueError('Box must have positive width and height')
        return self


class Book(Record):
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
    glare_fraction: Score | None = None
    dark_fraction: Score | None = None


class BookObservation(Record):
    sweep_id: Identifier
    book_id: Identifier
    frame_id: Identifier
    bbox: Box
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


class ReviewIssue(Record):
    sweep_id: Identifier
    book_id: Identifier
    field: str
    reason: Annotated[str, Field(min_length=1)]
    status: Literal['open', 'resolved'] = 'open'


class ProcessingJob(Record):
    sweep_id: Identifier
    frame_id: Identifier
    book_id: Identifier | None = None
    task: Literal['identify', 'measure', 'value', 'visual_diagnosis']
    status: Literal['pending', 'running', 'succeeded', 'failed', 'cancelled'] = 'pending'
    created_at: AwareDatetime
    model: str | None = None
    configuration_version: str = '1'
    result: dict[str, Any] | None = None
    error: str | None = None
