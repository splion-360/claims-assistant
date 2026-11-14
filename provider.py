"""Small Crusoe inference boundary shared by conversation, extraction, and vision."""
from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import TypeVar

import httpx
from dotenv import load_dotenv
from pydantic import BaseModel

from library_claim import telemetry
from library_claim.config import CONFIG

load_dotenv(Path(__file__).with_name('.env'))
MODEL = CONFIG.inference.model
VISION_MODEL = CONFIG.inference.vision_model
BASE_URL = os.getenv('CRUSOE_BASE_URL', 'https://api.inference.crusoecloud.com/v1').rstrip('/')
logger = logging.getLogger(__name__)
T = TypeVar('T', bound=BaseModel)


def strict_schema(schema: type[BaseModel]) -> dict:
    """Require explicit fields so model omissions cannot silently take defaults."""
    result = schema.model_json_schema()
    def visit(node):
        if isinstance(node, dict):
            if node.get('type') == 'object':
                node['additionalProperties'] = False
                node['required'] = list(node.get('properties', {}))
            node.pop('default', None)
            for value in node.values():
                visit(value)
        elif isinstance(node, list):
            for value in node:
                visit(value)
    visit(result)
    return result


class InferenceError(RuntimeError):
    """Safe, user-facing failure without credentials or provider response bodies."""


def has_api_key() -> bool:
    return bool(os.getenv('CRUSOE_API_KEY', '').strip())


def _models(selected: str) -> list[str]:
    """Order the models to try: the requested one, then its configured fallbacks.

    Args:
        selected: Model requested for this call.

    Returns:
        Distinct model IDs in the order they should be tried.
    """
    fallbacks = CONFIG.inference.vision_fallback_models if selected == VISION_MODEL else CONFIG.inference.fallback_models
    return list(dict.fromkeys([selected, *fallbacks]))


async def _attempt(client: httpx.AsyncClient, messages: list[dict], schema: type[BaseModel] | None,
                   model: str, max_tokens: int) -> str:
    """Send one completion request to one model.

    Args:
        client: Open HTTP client.
        messages: Provider-compatible conversation messages.
        schema: Optional structured response contract.
        model: Model to call.
        max_tokens: Output token budget.

    Returns:
        The nonempty model response text.

    Raises:
        InferenceError: Transport or the response is unusable.
    """
    payload = {'model': model, 'messages': messages,
               'max_tokens': max_tokens, 'temperature': CONFIG.inference.temperature}
    # Qwen models support this switch; other models keep their defaults.
    if model.startswith('Qwen/'):
        payload['chat_template_kwargs'] = {'enable_thinking': CONFIG.inference.qwen_enable_thinking}
    if schema:
        payload['response_format'] = {'type': 'json_schema', 'json_schema': {
            'name': schema.__name__, 'schema': strict_schema(schema), 'strict': True}}
    started = time.monotonic()
    try:
        response = await client.post(BASE_URL + '/chat/completions',
            headers={'Authorization': 'Bearer ' + os.environ['CRUSOE_API_KEY']}, json=payload)
        if response.status_code != 200:
            raise InferenceError(f'Crusoe request failed (HTTP {response.status_code}). Check model access, credits, or rate limits.')
        body = response.json()
        choice = body['choices'][0]
        if choice.get('finish_reason') == 'length':
            raise InferenceError('Crusoe response exceeded the output limit. Please retry with a shorter request.')
        content = choice['message'].get('content')
        if not isinstance(content, str) or not content.strip():
            raise InferenceError('Crusoe returned no usable answer. Please retry.')
        logger.info('Crusoe model=%s latency_s=%.2f usage=%s', model, time.monotonic() - started, body.get('usage', {}))
        return content
    except httpx.RequestError:
        raise InferenceError('Could not reach Crusoe. Please retry.') from None
    except (ValueError, KeyError, IndexError, TypeError):
        raise InferenceError('Crusoe returned an invalid response. Please retry.') from None


async def _run(messages: list[dict], schema: type[BaseModel] | None, model: str | None, max_tokens: int,
               validate=None):
    """Try the requested model, then each fallback, until one gives a usable answer.

    Args:
        messages: Provider-compatible conversation messages.
        schema: Optional structured response contract.
        model: Requested model, or None for the configured text model.
        max_tokens: Output token budget.
        validate: Optional callable turning response text into the result; may raise InferenceError.

    Returns:
        The response text, or the validated result.

    Raises:
        InferenceError: Credentials are missing or every model failed.
    """
    if not has_api_key():
        raise InferenceError('Set CRUSOE_API_KEY in .env and restart the server.')
    models = _models(model or MODEL)
    error = None
    async with httpx.AsyncClient(timeout=httpx.Timeout(CONFIG.inference.timeout_s, connect=CONFIG.inference.connect_timeout_s)) as client:
        for index, candidate in enumerate(models):
            if index:
                # Tell the logs and the flow view that a different model is answering.
                logger.warning('Crusoe model=%s failed (%s); retrying with %s', models[index - 1], error, candidate)
                telemetry.emit('fallback', 'done', f'{models[index - 1]} failed → {candidate}')
            try:
                text = await _attempt(client, messages, schema, candidate, max_tokens)
                return validate(text) if validate else text
            except InferenceError as failure:
                error = failure
    raise error


async def complete(messages: list[dict], *, schema: type[BaseModel] | None = None,
                   model: str | None = None, max_tokens: int = CONFIG.inference.max_tokens) -> str:
    """Request a completion, falling back to other configured models if Crusoe fails.

    Args:
        messages: Provider-compatible conversation messages.
        schema: Optional structured response contract.
        model: Optional per-call override of the configured text model.
        max_tokens: Output token budget, defaulting to config.yml.

    Returns:
        The nonempty model response text.

    Raises:
        InferenceError: Credentials are missing or every model failed.
    """
    return await _run(messages, schema, model, max_tokens)


async def structured(instruction: str, content: str | list[dict], schema: type[T],
                     *, model: str | None = None) -> T:
    """Request a validated structured answer, trying fallback models on failure or invalid data.

    Args:
        instruction: System instruction.
        content: User content, text or multimodal parts.
        schema: Pydantic model the answer must satisfy.
        model: Optional per-call override of the configured text model.

    Returns:
        The validated answer.

    Raises:
        InferenceError: Credentials are missing or every model failed.
    """
    def validate(text: str) -> T:
        """Parse a response, treating invalid data as a failure of that model.

        Args:
            text: Response text.

        Returns:
            The validated answer.

        Raises:
            InferenceError: The data failed validation.
        """
        try:
            return schema.model_validate_json(text)
        except ValueError:
            raise InferenceError('Crusoe returned data that failed validation. No result was saved.') from None

    return await _run([{'role': 'system', 'content': instruction}, {'role': 'user', 'content': content}],
                      schema, model, CONFIG.inference.max_tokens, validate)
