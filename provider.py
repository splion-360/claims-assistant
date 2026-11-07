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


async def complete(messages: list[dict], *, schema: type[BaseModel] | None = None,
                   model: str | None = None, max_tokens: int = CONFIG.inference.max_tokens) -> str:
    """Request a completion using centrally configured generation defaults.

    Args:
        messages: Provider-compatible conversation messages.
        schema: Optional structured response contract.
        model: Optional per-call override of the configured text model.
        max_tokens: Output token budget, defaulting to config.yml.

    Returns:
        The nonempty model response text.

    Raises:
        InferenceError: Credentials, transport, or the response are invalid.
    """
    if not has_api_key():
        raise InferenceError('Set CRUSOE_API_KEY in .env and restart the server.')
    selected_model = model or MODEL
    payload = {'model': selected_model, 'messages': messages,
               'max_tokens': max_tokens, 'temperature': CONFIG.inference.temperature}
    # The default Qwen model supports this switch; other models keep their defaults.
    if selected_model.startswith('Qwen/'):
        payload['chat_template_kwargs'] = {'enable_thinking': CONFIG.inference.qwen_enable_thinking}
    if schema:
        payload['response_format'] = {'type': 'json_schema', 'json_schema': {
            'name': schema.__name__, 'schema': strict_schema(schema), 'strict': True}}
    started = time.monotonic()
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(CONFIG.inference.timeout_s, connect=CONFIG.inference.connect_timeout_s)) as client:
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
        logger.info('Crusoe model=%s latency_s=%.2f usage=%s', selected_model,
                    time.monotonic() - started, body.get('usage', {}))
        return content
    except httpx.RequestError as exc:
        raise InferenceError('Could not reach Crusoe. Please retry.') from None
    except (ValueError, KeyError, IndexError, TypeError):
        raise InferenceError('Crusoe returned an invalid response. Please retry.') from None


async def structured(instruction: str, content: str | list[dict], schema: type[T],
                     *, model: str | None = None) -> T:
    text = await complete([{'role': 'system', 'content': instruction},
                           {'role': 'user', 'content': content}], schema=schema, model=model)
    try:
        return schema.model_validate_json(text)
    except ValueError:
        raise InferenceError('Crusoe returned data that failed validation. No result was saved.') from None
