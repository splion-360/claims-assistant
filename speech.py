"""Deepgram speech transport. Credentials stay on the server."""
from __future__ import annotations

import os
import re
from urllib.parse import urlencode

import httpx
from websockets.asyncio.client import connect

from library_claim.config import CONFIG


def speakable(text: str) -> str:
    """Turn model output into plain words, so markdown symbols are never read aloud or shown.

    Args:
        text: Reply or guidance text that may contain markdown.

    Returns:
        The same words without emphasis, headings, code marks, link targets or list bullets.
    """
    text = re.sub(r'\[([^\]]+)\]\([^)]*\)', r'\1', text)          # [label](url) -> label
    text = re.sub(r'(\*\*|__|\*|_|`)(\S(?:.*?\S)?)\1', r'\2', text)  # **x**, *x*, _x_, `x` -> x
    lines = []
    for line in text.splitlines():
        # Headings and list bullets lose their markers; each becomes its own sentence.
        line = re.sub(r'^\s*(#{1,6}\s+|[-*+]\s+|\d+[.)]\s+)', '', line).strip()
        if line:
            lines.append(line)
    text = ''
    for line in lines:
        text += line if not text else (' ' if text[-1] in '.!?:;,' else '. ') + line
    # Any stray symbol left over would be spoken as a word, e.g. "star".
    text = re.sub(r'[*#`]+', ' ', text)
    return re.sub(r'\s{2,}', ' ', text).strip()


class SpeechError(RuntimeError):
    """Safe speech failure suitable for displaying to the user."""


def enabled() -> bool:
    return bool(os.getenv('DEEPGRAM_API_KEY', '').strip())


def headers() -> dict[str, str]:
    if not enabled():
        raise SpeechError('Set DEEPGRAM_API_KEY in .env and restart to enable voice.')
    return {'Authorization': 'Token ' + os.environ['DEEPGRAM_API_KEY']}


def listen():
    """Configure a Deepgram stream using the selected speech recognition model.

    Returns:
        A websocket connection context manager for mono PCM16 at 16 kHz.

    Raises:
        SpeechError: The API credential is missing.
    """
    query = urlencode({'model': CONFIG.speech.stt_model,
                       'encoding': 'linear16', 'sample_rate': 16000})
    return connect('wss://api.deepgram.com/v2/listen?' + query,
                   additional_headers=headers(), open_timeout=CONFIG.speech.connect_timeout_s, close_timeout=CONFIG.speech.close_timeout_s,
                   max_size=1_000_000)


async def synthesize(text: str):
    """Synthesize speech using configured model, request size, and timeouts.

    Args:
        text: Full response to synthesize in sequential provider requests.

    Yields:
        Sample-aligned chunks of raw mono PCM16 at 24 kHz.

    Raises:
        SpeechError: Credentials, transport, or returned audio are invalid.
    """
    params = {'model': CONFIG.speech.tts_model,
              'encoding': 'linear16', 'sample_rate': 24000, 'container': 'none'}
    # Aura REST requests have a character limit; preserve all text across chunks.
    async with httpx.AsyncClient(timeout=httpx.Timeout(CONFIG.speech.timeout_s, connect=CONFIG.speech.connect_timeout_s)) as client:
        for start in range(0, len(text), CONFIG.speech.tts_chunk_chars):
            try:
                async with client.stream('POST', 'https://api.deepgram.com/v1/speak',
                                         headers=headers(), params=params,
                                         json={'text': text[start:start + CONFIG.speech.tts_chunk_chars]}) as response:
                    if response.status_code != 200:
                        raise SpeechError(f'Deepgram speech failed (HTTP {response.status_code}). Text is still available.')
                    pending = b''
                    async for chunk in response.aiter_bytes():
                        pending += chunk
                        length = len(pending) // 2 * 2
                        if length:
                            yield pending[:length]
                            pending = pending[length:]
                    if pending:
                        raise SpeechError('Deepgram returned incomplete audio. Text is still available.')
            except httpx.RequestError:
                raise SpeechError('Could not reach Deepgram. Text is still available.') from None
