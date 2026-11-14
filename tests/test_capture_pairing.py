"""Verify a phone pairs as a sweep's camera without gaining its other controls."""
import asyncio
import json
import time
import unittest
from types import SimpleNamespace as NS

import live_demo.server as s


class PairingCodeTests(unittest.TestCase):
    """Exercise issuing and resolving the short code a phone types."""

    def setUp(self):
        """Register one live sweep."""
        self.session = s.IntakeSession('pairing', owner='owner')
        s.sessions[self.session.session_id] = self.session
        self.addCleanup(s.sessions.pop, self.session.session_id, None)

    def test_code_is_six_digits_and_stable_while_current(self):
        """Issue one code per sweep and keep it until it expires."""
        code = s.capture_code_for(self.session)
        self.assertRegex(code, r'^[0-9]{6}$')
        self.assertEqual(s.capture_code_for(self.session), code)

    def test_code_resolves_to_its_own_sweep(self):
        """Find the sweep a current code belongs to, and no other."""
        code = s.capture_code_for(self.session)
        self.assertIs(s.session_for_code(code), self.session)
        self.assertIsNone(s.session_for_code('000000' if code != '000000' else '111111'))

    def test_expired_code_pairs_nothing(self):
        """Refuse a code once it has aged out."""
        code = s.capture_code_for(self.session)
        self.session.capture_code_at = time.monotonic() - s.CAPTURE_CODE_TTL - 1
        self.assertIsNone(s.session_for_code(code))

    def test_deleted_sweep_pairs_nothing(self):
        """Refuse a code belonging to a sweep that has been discarded."""
        code = s.capture_code_for(self.session)
        self.session.deleted = True
        self.assertIsNone(s.session_for_code(code))


class CaptureSocketTests(unittest.IsolatedAsyncioTestCase):
    """Exercise what a paired phone may and may not send."""

    def setUp(self):
        """Register a live sweep with an inbox the dispatch would read."""
        self.session = s.IntakeSession('paired', owner='owner')
        self.session.inbox = asyncio.Queue(maxsize=8)
        s.sessions[self.session.session_id] = self.session
        self.addCleanup(s.sessions.pop, self.session.session_id, None)

    def socket(self, code, messages):
        """Build a phone-side websocket double.

        Args:
            code: Pairing code sent in the query string.
            messages: Message dicts the phone sends before disconnecting.

        Returns:
            The socket double, recording what the server sent back.
        """
        outbound, queue = [], list(messages)

        class WS:
            headers = {'host': '127.0.0.1:4177'}
            client = NS(host='127.0.0.1')
            query_params = {'code': code}
            closed = None

            async def accept(self):
                """Accept the connection."""

            async def close(self, code=1000):
                """Record the close code.

                Args:
                    code: WebSocket close code.
                """
                WS.closed = code

            async def send_text(self, payload):
                """Record a server message.

                Args:
                    payload: JSON text sent to the phone.
                """
                outbound.append(json.loads(payload))

            async def receive_text(self):
                """Return the next queued phone message.

                Returns:
                    The message as JSON text.

                Raises:
                    WebSocketDisconnect: The phone has nothing left to send.
                """
                if not queue:
                    raise s.WebSocketDisconnect()
                return json.dumps(queue.pop(0))

        return WS(), outbound

    async def test_frames_reach_the_sweep_inbox(self):
        """Put a paired phone's camera messages where the live dispatch reads them."""
        code = s.capture_code_for(self.session)
        websocket, outbound = self.socket(code, [{'type': 'camera_state', 'enabled': True},
                                                 {'type': 'video', 'data': 'AAAA'}])
        await s.live_capture(websocket)
        self.assertEqual(outbound[0]['type'], 'paired')
        queued = [json.loads(self.session.inbox.get_nowait()) for _ in range(2)]
        self.assertEqual([item['type'] for item in queued], ['camera_state', 'video'])

    async def test_a_phone_cannot_talk_or_review(self):
        """Refuse every message kind that is not camera input."""
        code = s.capture_code_for(self.session)
        websocket, outbound = self.socket(code, [{'type': 'text', 'text': 'hello'},
                                                 {'type': 'audio', 'data': 'AAAA'},
                                                 {'type': 'book_capture', 'book_id': 'b1'}])
        await s.live_capture(websocket)
        self.assertTrue(self.session.inbox.empty())
        refusals = [item for item in outbound if item['type'] == 'error']
        self.assertEqual(len(refusals), 3)

    async def test_a_wrong_code_is_refused(self):
        """Close a connection whose code matches no sweep."""
        websocket, _ = self.socket('999999', [])
        self.session.capture_code = '123456'
        self.session.capture_code_at = time.monotonic()
        await s.live_capture(websocket)
        self.assertEqual(type(websocket).closed, 1008)

    async def test_only_one_phone_pairs_at_a_time(self):
        """Refuse a second camera while one is already paired."""
        code = s.capture_code_for(self.session)
        self.session.capture_socket = object()
        websocket, _ = self.socket(code, [])
        await s.live_capture(websocket)
        self.assertEqual(type(websocket).closed, 1008)

    async def test_a_sweep_with_no_live_browser_pairs_nothing(self):
        """Refuse pairing when no browser is running the sweep."""
        code = s.capture_code_for(self.session)
        self.session.inbox = None
        websocket, _ = self.socket(code, [])
        await s.live_capture(websocket)
        self.assertEqual(type(websocket).closed, 1008)

    async def test_a_full_inbox_drops_the_newest_frame(self):
        """Drop a frame rather than queue a backlog of stale views."""
        code = s.capture_code_for(self.session)
        self.session.inbox = asyncio.Queue(maxsize=1)
        websocket, outbound = self.socket(code, [{'type': 'video', 'data': 'AAAA'},
                                                 {'type': 'video', 'data': 'BBBB'}])
        await s.live_capture(websocket)
        self.assertEqual(self.session.inbox.qsize(), 1)
        self.assertNotIn('error', [item['type'] for item in outbound])

    async def test_pairing_releases_the_slot_on_disconnect(self):
        """Let another phone pair once the first one leaves."""
        code = s.capture_code_for(self.session)
        websocket, _ = self.socket(code, [])
        await s.live_capture(websocket)
        self.assertIsNone(self.session.capture_socket)
