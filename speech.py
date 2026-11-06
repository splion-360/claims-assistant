"""Deepgram speech transport. Credentials stay on the server."""
from __future__ import annotations

import os
from urllib.parse import urlencode

import httpx
from websockets.asyncio.client import connect


class SpeechError(RuntimeError):
    """Safe speech failure suitable for displaying to the user."""


def enabled() -> bool:
    return bool(os.getenv('DEEPGRAM_API_KEY', '').strip())


def headers() -> dict[str, str]:
    if not enabled():
        raise SpeechError('Set DEEPGRAM_API_KEY in .env and restart to enable voice.')
    return {'Authorization': 'Token ' + os.environ['DEEPGRAM_API_KEY']}


def listen():
    query = urlencode({'model': os.getenv('DEEPGRAM_STT_MODEL', 'flux-general-en'),
                       'encoding': 'linear16', 'sample_rate': 16000})
    return connect('wss://api.deepgram.com/v2/listen?' + query,
                   additional_headers=headers(), open_timeout=10, close_timeout=2,
                   max_size=1_000_000)


async def synthesize(text: str):
    """Yield raw mono PCM16 at 24 kHz, keeping each chunk sample-aligned."""
    params = {'model': os.getenv('DEEPGRAM_TTS_MODEL', 'aura-2-thalia-en'),
              'encoding': 'linear16', 'sample_rate': 24000, 'container': 'none'}
    # Aura REST requests have a character limit; preserve all text across chunks.
    async with httpx.AsyncClient(timeout=httpx.Timeout(30, connect=10)) as client:
        for start in range(0, len(text), 1800):
            try:
                async with client.stream('POST', 'https://api.deepgram.com/v1/speak',
                                         headers=headers(), params=params,
                                         json={'text': text[start:start + 1800]}) as response:
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
