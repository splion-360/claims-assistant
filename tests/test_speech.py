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
