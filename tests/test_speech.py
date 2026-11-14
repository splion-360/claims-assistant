import json
import unittest
from unittest.mock import patch
import httpx
import speech


class SpeechTests(unittest.IsolatedAsyncioTestCase):
    async def test_tts_uses_server_key_and_pcm_format(self):
        def handler(request):
            self.assertEqual(request.headers['Authorization'],'Token test-key')
            self.assertEqual(request.url.params['encoding'],'linear16')
            self.assertEqual(request.url.params['container'],'none')
            self.assertEqual(request.url.params['sample_rate'],'24000')
            self.assertEqual(json.loads(request.content)['text'],'Hello')
            return httpx.Response(200,content=b'\0\0\1\0')
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler))
        with patch.dict('os.environ',{'DEEPGRAM_API_KEY':'test-key'}),patch.object(speech.httpx,'AsyncClient',return_value=client):
            audio=b''.join([part async for part in speech.synthesize('Hello')])
        self.assertEqual(audio,b'\0\0\1\0')

    async def test_error_does_not_expose_response_or_key(self):
        client=httpx.AsyncClient(transport=httpx.MockTransport(lambda request:httpx.Response(401,text='secret')))
        with patch.dict('os.environ',{'DEEPGRAM_API_KEY':'secret'}),patch.object(speech.httpx,'AsyncClient',return_value=client):
            with self.assertRaises(speech.SpeechError) as caught:
                _=[part async for part in speech.synthesize('Hello')]
        self.assertNotIn('secret',str(caught.exception))

    def test_listen_requires_key(self):
        with patch.dict('os.environ',{'DEEPGRAM_API_KEY':''}):
            with self.assertRaises(speech.SpeechError):speech.listen()


class SpeakableTextTests(unittest.TestCase):
    """Markdown from a model never reaches the speech engine as symbols."""

    def test_markdown_is_removed_but_words_are_kept(self):
        """Emphasis, headings, code, links and bullets become plain words."""
        from speech import speakable
        cases = {
            'The author for *Hawkins Pressure Cookery* is currently unknown.':
                'The author for Hawkins Pressure Cookery is currently unknown.',
            '## Found **The Good People** by _Hannah Kent_': 'Found The Good People by Hannah Kent',
            'Try `steady` on [Dune](https://example.org) now.': 'Try steady on Dune now.',
            '- first book\n- second book': 'first book. second book',
            "It's a well-known 5 * 3 shelf": "It's a well-known 5 3 shelf",
        }
        for raw, spoken in cases.items():
            with self.subTest(raw=raw):
                self.assertEqual(speakable(raw), spoken)
