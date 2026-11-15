"""Hold sourced prices and the evidence that backs each one."""
from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

Amount = Annotated[float, Field(ge=0, allow_inf_nan=False)]


class Price(BaseModel):
    """A money figure together with the source an adjuster can retrieve."""
    model_config = ConfigDict(extra='forbid', frozen=True)
    amount: Amount
    currency: str
    source: str
    url: str
    retrieved_at: datetime
    # True when the figure was converted from another market's currency.
    converted: bool = False
    converted_from: str | None = None
    fx_rate: Amount | None = None
    fx_source: str | None = None
    condition_assumed: str = ''
    kind: Literal['replacement', 'used'] = 'replacement'

    def as_replacement_block(self) -> dict:
        """Render the packet's replacement cost block.

        Returns:
            The block the output contract fixes for a replacement cost.
        """
        return {'amount': round(self.amount, 2), 'currency': self.currency, 'source': self.source,
                'url': self.url, 'retrieved_at': self.retrieved_at.isoformat(), 'converted': self.converted,
                'converted_from': self.converted_from, 'fx_rate': self.fx_rate, 'fx_source': self.fx_source}

    def as_used_block(self) -> dict:
        """Render the packet's used value block.

        Returns:
            The block the output contract fixes for a used value.
        """
        return {'amount': round(self.amount, 2), 'currency': self.currency, 'source': self.source,
                'url': self.url, 'retrieved_at': self.retrieved_at.isoformat(),
                'condition_assumed': self.condition_assumed, 'converted': self.converted,
                'converted_from': self.converted_from}


class Valuation(BaseModel):
    """The pricing outcome for one book."""
    model_config = ConfigDict(extra='forbid', frozen=True)
    book_id: str
    country: str
    currency: str
    replacement: Price | None = None
    # Upper end of a sourced range for an object whose brand and model are unknown.
    replacement_high: Price | None = None
    used: Price | None = None
    needs_appraisal: bool = False
    # Why a figure is missing or why the book was withheld from automatic pricing.
    reasons: tuple[str, ...] = ()
    # True when a source timed out or refused for now, so the lookup is worth repeating.
    retry_later: bool = False
