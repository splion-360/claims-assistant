"""Send typed decision questions to the Drex System One endpoint."""
import base64
import logging
import os
import time
from pathlib import Path
from typing import Literal

import httpx
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from library_claim.config import CONFIG, ROOT

load_dotenv(Path(ROOT) / '.env')
logger = logging.getLogger(__name__)
# Statuses where one retry is worthwhile; 401/402/422 cannot succeed on retry.
RETRY_STATUSES = {429, 500, 529}


class DrexError(RuntimeError):
    """Safe decision failure that never contains credentials or response bodies."""

    def __init__(self, message: str, status: int | None = None):
        """Store a safe message and the HTTP status when one was received.

        Args:
            message: User-safe description of the failure.
            status: HTTP status returned by Drex, if any.
        """
        super().__init__(message)
        self.status = status


class Answer(BaseModel):
    """One typed answer; fields not used by the question type stay empty."""
    type: Literal['noul', 'choice', 'score']
    noul: float | None = None
    choice: str | None = None
    probabilities: dict[str, float] = Field(default_factory=dict)
    confidence: float | None = None


class DrexResult(BaseModel):
    """Answers plus the provenance needed to trace and cost a decision."""
    model: str
    answers: dict[str, Answer]
    request_id: str | None = None
    input_tokens: int = 0
    latency_ms: float


def noul(instructions: str) -> dict:
    """Build a yes/no question.

    Args:
        instructions: The question Drex answers with a probability of yes.

    Returns:
        A System One question object.
    """
    return {'type': 'noul', 'instructions': instructions}


def choice(instructions: str, criteria: dict[str, str]) -> dict:
    """Build a multiple-choice question.

    Args:
        instructions: The question to answer.
        criteria: Option labels mapped to their descriptions.

    Returns:
        A System One question object.
    """
    return {'type': 'choice', 'instructions': instructions, 'criteria': criteria}


def _post(client: httpx.Client, payload: dict, key: str) -> httpx.Response:
    """Send one request, retrying once for rate limits and transient server errors.

    Args:
        client: HTTP client to use.
        payload: System One request body.
        key: Drex API key.

    Returns:
        The final HTTP response.

    Raises:
        DrexError: Drex could not be reached.
    """
    headers = {'Authorization': 'Bearer ' + key, 'X-Nace-Client': CONFIG.drex.client_name}
    url = CONFIG.drex.base_url.rstrip('/') + '/v1/systemone'
    for attempt in range(2):
        try:
            response = client.post(url, json=payload, headers=headers)
        except httpx.RequestError:
            raise DrexError('Could not reach Drex.') from None
        if response.status_code not in RETRY_STATUSES or attempt:
            return response
        # Drex states how long to back off; default to one second when it does not.
        try:
            delay = float(response.headers.get('retry-after-ms', '')) / 1000
        except ValueError:
            delay = 1.0
        time.sleep(min(max(delay, 0), 10))
    return response


def decide(questions: dict[str, dict], *, state: object, images: list[bytes] | None = None,
           client: httpx.Client | None = None) -> DrexResult:
    """Ask Drex typed questions about a state and optional JPEG images.

    Args:
        questions: Question IDs mapped to question objects from `noul`/`choice`.
        state: Any JSON-serialisable description of the situation.
        images: JPEG bytes; Drex Max accepts at most eight.
        client: Optional HTTP client, used by tests.

    Returns:
        Typed answers with model, request ID, input tokens and latency.

    Raises:
        DrexError: Credentials, transport, or the response are unusable.
    """
    key = os.getenv('DREX_API_KEY', '').strip()
    if not key:
        raise DrexError('Set DREX_API_KEY in .env and restart the server.')
    payload = {'model': CONFIG.drex.model, 'state': state, 'questions': questions}
    if images:
        payload['images'] = ['data:image/jpeg;base64,' + base64.b64encode(image).decode() for image in images]
    owned = client is None
    client = client or httpx.Client(timeout=httpx.Timeout(CONFIG.drex.timeout_s, connect=CONFIG.drex.connect_timeout_s))
    started = time.monotonic()
    try:
        response = _post(client, payload, key)
    finally:
        if owned:
            client.close()
    if response.status_code != 200:
        raise DrexError(f'Drex request failed (HTTP {response.status_code}).', response.status_code)
    try:
        body = response.json()
        result = DrexResult(model=body['model'], answers=body['answers'], request_id=body.get('request_id'),
                            input_tokens=body.get('usage', {}).get('input_tokens', 0),
                            latency_ms=(time.monotonic() - started) * 1000)
    except (ValueError, KeyError, TypeError, AttributeError):
        raise DrexError('Drex returned an invalid response.', response.status_code) from None
    # Every requested question must come back with a value of its own type.
    for question_id, question in questions.items():
        answer = result.answers.get(question_id)
        value = None if answer is None else getattr(answer, question['type'], None)
        if answer is None or answer.type != question['type'] or value is None:
            raise DrexError('Drex returned an incomplete decision.', response.status_code)
    logger.info('Drex request_id=%s latency_ms=%.0f input_tokens=%d', result.request_id,
                result.latency_ms, result.input_tokens)
    return result
