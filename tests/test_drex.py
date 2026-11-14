"""Verify the Drex decision client contract without network calls."""
import json
import os
import unittest
from unittest.mock import patch

import httpx

from library_claim.decisions.drex import DrexError, choice, decide, noul

SMOKE_RESPONSE = {
    'model': 'drex-max-v1.1',
    'answers': {
        'book_visible': {'type': 'noul', 'noul': 0.002, 'confidence': 0.998},
        'issue': {'type': 'choice', 'choice': 'no_books', 'confidence': 0.98,
                  'probabilities': {'none': 0.01, 'no_books': 0.99}},
    },
    'usage': {'input_tokens': 803, 'output_tokens': 298, 'image_tokens': 576},
    'evaluation_time_ms': 445.4,
    'request_id': 'req_x',
}


def client_for(handler) -> httpx.Client:
    """Build an httpx client that routes every request to a test handler.

    Args:
        handler: Callable receiving the request and returning a response.

    Returns:
        A client backed by a mock transport.
    """
    return httpx.Client(transport=httpx.MockTransport(handler))


class DrexClientTests(unittest.TestCase):
    """Exercise request shape, parsing, retries, and safe failures."""

    def setUp(self):
        """Provide a test key for every request."""
        patcher = patch.dict(os.environ, {'DREX_API_KEY': 'test-key'})
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_decide_sends_images_and_parses_answers(self):
        """Send JPEGs as data URLs and expose typed answers and usage."""
        def handler(request):
            body = json.loads(request.content)
            self.assertEqual(str(request.url), 'https://console.nace.ai/v1/systemone')
            self.assertEqual(request.headers['Authorization'], 'Bearer test-key')
            self.assertEqual(request.headers['X-Nace-Client'], 'library-claim/1')
            self.assertEqual(body['model'], 'drex-max-v1.1')
            self.assertTrue(body['images'][0].startswith('data:image/jpeg;base64,'))
            self.assertEqual(body['questions']['book_visible']['type'], 'noul')
            self.assertEqual(body['questions']['issue']['criteria'], {'none': 'clear', 'no_books': 'empty'})
            return httpx.Response(200, json=SMOKE_RESPONSE)

        result = decide({'book_visible': noul('Is a book visible?'),
                         'issue': choice('What limits reading?', {'none': 'clear', 'no_books': 'empty'})},
                        state='frame', images=[b'\xff\xd8\xffjpeg'], client=client_for(handler))
        self.assertEqual(result.answers['book_visible'].noul, 0.002)
        self.assertEqual(result.answers['issue'].choice, 'no_books')
        self.assertEqual(result.answers['issue'].probabilities['no_books'], 0.99)
        self.assertEqual(result.input_tokens, 803)
        self.assertEqual(result.request_id, 'req_x')

    def test_rate_limit_retries_once_after_retry_after(self):
        """Honour retry-after-ms once for a rate-limited request."""
        calls = []

        def handler(request):
            calls.append(request)
            if len(calls) == 1:
                return httpx.Response(429, headers={'retry-after-ms': '1'}, json={'error': {}})
            return httpx.Response(200, json=SMOKE_RESPONSE)

        result = decide({'book_visible': noul('Is a book visible?')}, state='frame', client=client_for(handler))
        self.assertEqual(len(calls), 2)
        self.assertEqual(result.model, 'drex-max-v1.1')

    def test_payment_required_raises_safe_error(self):
        """Fail without retrying or echoing the provider body when credit is exhausted."""
        calls = []

        def handler(request):
            calls.append(request)
            return httpx.Response(402, json={'error': {'message': 'secret details'}})

        with self.assertRaises(DrexError) as raised:
            decide({'q': noul('?')}, state='x', client=client_for(handler))
        self.assertNotIn('secret', str(raised.exception))
        self.assertEqual(raised.exception.status, 402)
        self.assertEqual(len(calls), 1)

    def test_missing_key_raises(self):
        """Refuse to call Drex without credentials."""
        with patch.dict(os.environ, {'DREX_API_KEY': ''}):
            with self.assertRaises(DrexError):
                decide({'q': noul('?')}, state='x', client=client_for(lambda request: httpx.Response(200)))

    def test_malformed_answers_raise_drex_error(self):
        """Treat a 200 response with missing or mistyped answers as a failed decision."""
        bad_bodies = [
            {'model': 'm', 'answers': {}},
            {'model': 'm', 'answers': {'q': {'type': 'noul', 'noul': None}}},
            {'model': 'm', 'answers': {'q': {'type': 'choice', 'choice': 'x'}}},
        ]
        for body in bad_bodies:
            with self.subTest(body=body):
                with self.assertRaises(DrexError):
                    decide({'q': noul('?')}, state='x', client=client_for(lambda request, body=body: httpx.Response(200, json=body)))

    def test_unparseable_retry_after_still_retries(self):
        """Fall back to a default wait when retry-after-ms is not a number."""
        calls = []

        def handler(request):
            calls.append(request)
            if len(calls) == 1:
                return httpx.Response(429, headers={'retry-after-ms': 'soon'})
            return httpx.Response(200, json=SMOKE_RESPONSE)

        with patch('library_claim.decisions.drex.time.sleep'):
            decide({'book_visible': noul('?')}, state='x', client=client_for(handler))
        self.assertEqual(len(calls), 2)

    def test_decision_reports_stage_and_answer_summary(self):
        """A staged decision tells the flow view what Drex answered."""
        from library_claim import telemetry
        events = []
        with telemetry.listening(events.append):
            decide({'book_visible': noul('?'), 'issue': choice('?', {'none': 'a', 'no_books': 'b'})}, state='x',
                   stage='d1', client=client_for(lambda request: httpx.Response(200, json=SMOKE_RESPONSE)))
        self.assertEqual([(e['stage'], e['phase']) for e in events], [('d1', 'start'), ('d1', 'done')])
        self.assertEqual(events[1]['detail'], 'book_visible 0.00 · issue no_books 0.99')
