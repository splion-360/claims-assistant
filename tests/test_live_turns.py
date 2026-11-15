"""Text/camera transport stays responsive while inference is in flight."""
import asyncio
import contextlib
import base64
import json
import tempfile
from pathlib import Path
import unittest
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, Mock, patch
from live_demo import server as s


def unavailable_check(jpeg):
    """Fail every D1 decision the way an unreachable Drex would.

    Args:
        jpeg: Frame bytes.

    Raises:
        DrexError: Always.
    """
    from library_claim.decisions.drex import DrexError
    raise DrexError('Could not reach Drex.')


def title_prompt(signals):
    """Choose read_title whenever a visible book lacks a title, like the old fixed rule.

    Args:
        signals: D6 guidance signals.

    Returns:
        A guidance label.
    """
    return 'read_title' if signals.missing_title_books else 'stay_quiet'


class LiveTurnTests(unittest.IsolatedAsyncioTestCase):
    async def drive(self, receive, *, reply=None, frame_check=None, prompt=None,
                    library=False):
        """Exercise live transport with isolated evidence storage.

        Args:
            receive: Async function supplying simulated browser messages.
            reply: Optional conversation implementation.
            frame_check: Optional D1 decision double; by default every check fails.
            prompt: Optional D6 decision double; by default it asks for missing titles only.
            library: Start the session in library-sweep mode, as the app does.

        Returns:
            The session and emitted transport messages.
        """
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        session = s.IntakeSession('transport', owner='owner', library_sweep=library)
        s.sessions[session.session_id] = session
        self.addCleanup(s.sessions.pop, session.session_id, None)
        sent = []
        class WS:
            headers = {'host':'127.0.0.1:4177','origin':'http://127.0.0.1:4177'}
            client = NS(host='127.0.0.1')
            url = NS(scheme='ws')
            cookies = {'intake_owner':'owner'}
            query_params = {'session_id':'transport'}
            async def accept(self): pass
            async def close(self, **kwargs): pass
            async def send_json(self, payload): sent.append(payload)
            async def receive_text(self): return json.dumps(await receive(session, sent))
        with patch.object(s, 'check_frame', side_effect=frame_check or unavailable_check), \
             patch.object(s, 'choose_prompt', side_effect=prompt or title_prompt), \
             patch.object(s, 'resolve_book'), \
             patch.dict('os.environ', {'DREX_API_KEY': ''}), \
             patch.object(s, 'CAPTURE_DATA_DIR', Path(directory.name)), \
             patch.object(s, '_has_api_key', return_value=True), \
             patch.object(s, 'complete', side_effect=reply or AsyncMock(return_value='What happened?')):
            await asyncio.wait_for(s.live_voice(WS()), 3)
        return session, sent

    async def test_text_updates_packet_and_replies(self):
        count=0
        async def receive(session,sent):
            nonlocal count
            count+=1
            if count==1:return {'type':'text','text':'Hello','id':'one'}
            while not any(m['type']=='transcript' and m['speaker']=='Agent' for m in sent):
                await asyncio.sleep(.001)
            return {'type':'close'}
        session,sent=await self.drive(receive)
        self.assertEqual([t['text'] for t in session.transcript],['Hello','What happened?'])
        self.assertTrue(any(m['type']=='state' for m in sent))
        self.assertFalse([m for m in sent if m['type']=='processing'][-1]['active'])

    async def test_slow_reply_does_not_block_camera_or_close(self):
        started=asyncio.Event();count=0
        async def reply(*args,**kwargs):
            started.set();await asyncio.Event().wait()
        async def receive(session,sent):
            nonlocal count
            count+=1
            if count==1:return {'type':'text','text':'Test claim','id':'one'}
            if count==2:
                await started.wait()
                return {'type':'video','data':base64.b64encode(b'\xff\xd8\xffTEST').decode()}
            self.assertEqual(session.last_frame,b'\xff\xd8\xffTEST')
            return {'type':'close'}
        session,_=await self.drive(receive,reply=reply)
        self.assertIsNone(session.live_socket)
        self.assertIsNone(session.last_frame)
        self.assertFalse(session.tasks)

    async def test_new_text_cancels_stale_reply_and_preserves_correction(self):
        started=asyncio.Event();count=0;calls=0
        async def reply(*args,**kwargs):
            nonlocal calls
            calls+=1
            if calls==1:
                started.set();await asyncio.Event().wait()
            return 'Correction noted.'
        async def receive(session,sent):
            nonlocal count
            count+=1
            if count==1:return {'type':'text','text':'First version','id':'one'}
            if count==2:
                await started.wait()
                return {'type':'text','text':'Correction','id':'two'}
            while not any(m['type']=='transcript' and m['speaker']=='Agent' for m in sent):await asyncio.sleep(.001)
            return {'type':'close'}
        session,sent=await self.drive(receive,reply=reply)
        self.assertEqual([t['text'] for t in session.transcript],['First version','Correction','Correction noted.'])

    async def test_voice_rejected_and_camera_state_is_not_a_claimant_fact(self):
        messages=iter([{'type':'camera_state','enabled':True},{'type':'audio','data':'AAA='},{'type':'camera_state','enabled':False},{'type':'close'}])
        async def receive(*args):return next(messages)
        session,sent=await self.drive(receive)
        self.assertEqual(session.transcript,[])
        self.assertTrue(any(m['type']=='error' and 'voice' in m['message'] for m in sent))

    async def test_failed_camera_analysis_can_retry_the_same_frame(self):
        import cv2
        from unittest.mock import Mock
        from test_tracking import textured_view
        from test_frame_check import check

        _, image = cv2.imencode('.jpg', textured_view())
        jpeg = image.tobytes()
        count=0
        async def receive(session,sent):
            nonlocal count
            count+=1
            if count==1:return {'type':'video','data':base64.b64encode(jpeg).decode()}
            if count==2:
                while not any(m['type']=='error' for m in sent):await asyncio.sleep(.001)
                clock[0]+=9
                return {'type':'video','data':base64.b64encode(jpeg).decode()}
            while len([m for m in sent if m['type']=='error'])<2:await asyncio.sleep(.001)
            return {'type':'close'}
        clock=[100.0]
        # Override only server timing; do not change the asyncio event loop clock.
        with patch.object(s,'time',NS(monotonic=lambda:clock[0])),patch.object(s,'_pin_evidence_photo',side_effect=s.InferenceError('Temporary failure')) as capture:
            await self.drive(receive, frame_check=lambda jpeg: check(0.9))
        self.assertEqual(capture.call_count,2)

    async def test_spoken_turn_dispatches_once_and_returns_audio(self):
        events=asyncio.Queue()
        events.put_nowait(json.dumps({'type':'Connected'}))
        class Connection:
            async def __aenter__(self):return self
            async def __aexit__(self,*args):pass
            def __aiter__(self):return self
            async def __anext__(self):return await events.get()
            async def send(self,data):
                self.last_audio=data
                for name in ['StartOfTurn','EndOfTurn','EndOfTurn']:
                    events.put_nowait(json.dumps({'type':'TurnInfo','event':name,'turn_index':0,'transcript':'No injuries.'}))
        connection=Connection()
        async def audio(text):yield b'\0\0\1\0'
        count=0
        async def receive(session,sent):
            nonlocal count
            count+=1
            if count==1:return {'type':'voice_state','enabled':True}
            if count==2:
                while not any(m['type']=='voice_state' and m['enabled'] for m in sent):await asyncio.sleep(.001)
                return {'type':'audio','data':'AAA='}
            while not any(m['type']=='audio_end' for m in sent):await asyncio.sleep(.001)
            return {'type':'close'}
        with patch.object(s.speech,'enabled',return_value=True),patch.object(s.speech,'listen',return_value=connection),patch.object(s.speech,'synthesize',side_effect=audio):
            session,sent=await self.drive(receive)
        self.assertEqual([t['text'] for t in session.transcript],['No injuries.','What happened?'])
        self.assertTrue(any(m['type']=='audio' for m in sent))
        self.assertFalse(session.tasks)


class CorrectionReplyTests(unittest.IsolatedAsyncioTestCase):
    """The reply model hears the real correction outcome, and failures still get a reply."""

    async def sweep_turn(self, **patches):
        """Send one typed turn during a library sweep and return the reply model's messages.

        Args:
            **patches: Server attributes to replace for this turn.

        Returns:
            The messages passed to the reply model.
        """
        seen = []

        async def reply(messages, **kwargs):
            """Capture the reply request.

            Args:
                messages: Conversation messages.
                **kwargs: Generation options.

            Returns:
                A fixed reply.
            """
            seen.append(messages)
            return 'Okay.'

        count = 0

        async def receive(session, sent):
            """Send one turn and close after the agent replies.

            Args:
                session: Live session.
                sent: Emitted messages.

            Returns:
                Next simulated browser message.
            """
            nonlocal count
            count += 1
            if count == 1:
                session.library_sweep = True
                return {'type': 'text', 'text': "skip that shelf, those aren't mine", 'id': 'turn1'}
            while not any(m['type'] == 'transcript' and m['speaker'] == 'Agent' for m in sent):
                await asyncio.sleep(.001)
            return {'type': 'close'}

        with patch.multiple(s, **patches) if patches else contextlib.nullcontext():
            await LiveTurnTests.drive(self, receive, reply=reply)
        return seen

    async def test_failed_decision_tells_model_nothing_was_recorded(self):
        """With Drex unavailable, the reply context says no correction was applied."""
        seen = await self.sweep_turn()
        self.assertIn('No correction was applied this turn', seen[0][0]['content'])

    async def test_unexpected_correction_error_still_replies(self):
        """An unexpected failure in the correction path falls back to a normal reply."""
        def broken(text):
            """Fail unexpectedly.

            Args:
                text: Utterance.

            Raises:
                RuntimeError: Always.
            """
            raise RuntimeError('database locked')

        seen = await self.sweep_turn(classify_intent=broken)
        self.assertEqual(len(seen), 1)


class VoiceTurnStateTests(unittest.IsolatedAsyncioTestCase):
    """An unfinished spoken turn must not mute guidance after voice stops."""

    async def test_stopping_voice_mid_turn_clears_user_turn(self):
        """Reset the open-turn flag when voice stops before EndOfTurn arrives."""
        events = asyncio.Queue()
        events.put_nowait(json.dumps({'type': 'Connected'}))

        class Connection:
            """Deepgram stand-in that starts a turn and never ends it."""
            async def __aenter__(self): return self
            async def __aexit__(self, *args): pass
            def __aiter__(self): return self
            async def __anext__(self): return await events.get()
            async def send(self, data):
                events.put_nowait(json.dumps({'type': 'TurnInfo', 'event': 'StartOfTurn', 'turn_index': 0,
                                              'transcript': 'skip'}))

        count = 0

        async def receive(session, sent):
            """Start voice, open a turn, then stop voice mid-turn.

            Args:
                session: Live session.
                sent: Emitted messages.

            Returns:
                Next simulated browser message.
            """
            nonlocal count
            count += 1
            if count == 1:
                return {'type': 'voice_state', 'enabled': True}
            if count == 2:
                while not any(m['type'] == 'voice_state' and m['enabled'] for m in sent):
                    await asyncio.sleep(.001)
                return {'type': 'audio', 'data': 'AAA='}
            if count == 3:
                while not session.user_turn_open:
                    await asyncio.sleep(.001)
                return {'type': 'voice_state', 'enabled': False}
            await asyncio.sleep(0.05)
            return {'type': 'close'}

        with patch.object(s.speech, 'enabled', return_value=True), patch.object(s.speech, 'listen', return_value=Connection()):
            session, _ = await LiveTurnTests.drive(self, receive)
        self.assertFalse(session.user_turn_open)


class FlowViewTests(unittest.IsolatedAsyncioTestCase):
    """The /flow page is served and live sessions report their steps to it."""

    async def test_live_turn_reports_reply_stage(self):
        """A typed turn shows up on the flow bus as a finished Qwen reply."""
        s.FLOW.recent.clear()
        count = 0

        async def receive(session, sent):
            """Send one turn and close after the reply.

            Args:
                session: Live session.
                sent: Emitted messages.

            Returns:
                Next simulated browser message.
            """
            nonlocal count
            count += 1
            if count == 1:
                return {'type': 'text', 'text': 'Hello', 'id': 'flow1'}
            while not any(m['type'] == 'transcript' and m['speaker'] == 'Agent' for m in sent):
                await asyncio.sleep(.001)
            await asyncio.sleep(0)
            return {'type': 'close'}

        await LiveTurnTests.drive(self, receive)
        stages = [(event['stage'], event['phase']) for event in s.FLOW.recent]
        self.assertIn(('reply', 'start'), stages)
        self.assertIn(('reply', 'done'), stages)

    def test_flow_page_is_served(self):
        """GET /flow returns the flow view."""
        from fastapi.testclient import TestClient
        response = TestClient(s.app, base_url='http://127.0.0.1:4177').get('/flow')
        self.assertEqual(response.status_code, 200)
        self.assertIn('id="graph"', response.text)
        self.assertIn('id="log"', response.text)


class ReadLimitTests(unittest.IsolatedAsyncioTestCase):
    """Spine reads continue past 20 per session while kept photos stay bounded."""

    async def test_reads_continue_past_twenty_and_ui_keeps_recent_photos(self):
        """25 reads all succeed; only the most recent 20 photos and notes are kept for the UI."""
        import time as clock
        session = s.IntakeSession('reads', owner='owner')
        observation = s.FrameObservation(books=[], complete=True, observation='shelf', supports_claimant_description=False)
        with patch.object(s, 'structured', AsyncMock(return_value=observation)):
            for index in range(25):
                session.last_frame, session.last_frame_at, session.last_frame_id = b'\xff\xd8\xff', clock.monotonic(), f'f{index}'
                result = await s._pin_evidence_photo(session, {})
                self.assertTrue(result['pinned'], result)
        self.assertEqual(session.read_count, 25)
        self.assertEqual(len(session.evidence_photos), 20)
        self.assertEqual(session.evidence_photos[-1]['frame_id'], 'f24')
        self.assertEqual(len(session.camera_notes), 20)

    async def test_configured_read_limit_stops_reading(self):
        """The safety limit from config stops further reads with a neutral message."""
        import time as clock
        session = s.IntakeSession('limit', owner='owner')
        session.read_count = s.CONFIG.capture.max_reads_per_session
        session.last_frame, session.last_frame_at, session.last_frame_id = b'\xff\xd8\xff', clock.monotonic(), 'f'
        result = await s._pin_evidence_photo(session, {})
        self.assertFalse(result['pinned'])
        self.assertIn('limit', result['message'])


class PlainReplyTests(unittest.IsolatedAsyncioTestCase):
    """Replies are shown and spoken without markdown symbols."""

    async def test_markdown_reply_is_cleaned_before_display_and_speech(self):
        """A model reply with *emphasis* reaches the transcript as plain words."""
        count = 0

        async def receive(session, sent):
            """Send one turn and close after the reply.

            Args:
                session: Live session.
                sent: Emitted messages.

            Returns:
                Next simulated browser message.
            """
            nonlocal count
            count += 1
            if count == 1:
                return {'type': 'text', 'text': 'Who wrote it?', 'id': 'md1'}
            while not any(m['type'] == 'transcript' and m['speaker'] == 'Agent' for m in sent):
                await asyncio.sleep(.001)
            return {'type': 'close'}

        reply = AsyncMock(return_value='The author for *Hawkins Pressure Cookery* is currently unknown.')
        session, _ = await LiveTurnTests.drive(self, receive, reply=reply)
        self.assertEqual(session.transcript[-1]['text'], 'The author for Hawkins Pressure Cookery is currently unknown.')


class SweepConversationTests(unittest.IsolatedAsyncioTestCase):
    """The agent opens by confirming the market, and reads short answers to its own questions."""

    async def test_greeting_then_country_answer_sets_the_market(self):
        """The greeting asks for the country; "the UK" is confirmed in code without a model reply."""
        count, database = 0, None

        async def receive(session, sent):
            """Answer the greeting with a country, then close after the confirmation."""
            nonlocal count, database
            count += 1
            if count == 1:
                database = s.CAPTURE_DATA_DIR / 'claims.sqlite3'
                return {'type': 'text', 'text': "I'm in the UK", 'id': 'c1'}
            while len(session.transcript) < 3:
                await asyncio.sleep(.001)
            return {'type': 'close'}

        reply = AsyncMock(return_value='model reply')
        session, _ = await LiveTurnTests.drive(self, receive, reply=reply, library=True)
        self.assertIn('Is that right, or which country are you in?', session.transcript[0]['text'])
        self.assertEqual(session.country, 'GB')
        self.assertIn('British pounds', session.transcript[2]['text'])
        reply.assert_not_called()
        from library_claim.domain.models import Sweep
        from library_claim.storage.repository import Repository
        # The harness has restored the real data folder by now; read the test's own database.
        with Repository(database) as repo:
            self.assertEqual(repo.get(Sweep, 'transport').currency, 'GBP')

    async def test_print_answer_releases_the_piece(self):
        """An answer to "original or print?" is recorded on the object."""
        from datetime import datetime, timezone
        from library_claim.domain.models import Item, Sweep
        from library_claim.storage.repository import Repository
        count, database = 0, None

        async def receive(session, sent):
            """Ask about a saved portrait, answer "a print", then close."""
            nonlocal count, database
            count += 1
            if count == 1:
                database = s.CAPTURE_DATA_DIR / 'claims.sqlite3'
                with Repository(database) as repo:
                    repo.add_many([Sweep(id='transport', captured_at=datetime.now(timezone.utc), country='US'),
                                   Item(id='art1', sweep_id='transport', category='art', needs_appraisal=True)])
                session.country = 'US'
                session.pending_art = 'art1'
                return {'type': 'text', 'text': "It's just a print", 'id': 'p1'}
            while not any(t['text'].startswith('Noted') for t in session.transcript):
                await asyncio.sleep(.001)
            return {'type': 'close'}

        session, _ = await LiveTurnTests.drive(self, receive, library=True)
        self.assertIsNone(session.pending_art)
        with Repository(database) as repo:
            self.assertTrue(repo.get(Item, 'art1').stated_print)

    async def test_camera_stop_reads_the_summary_and_records_time_to_packet(self):
        """Stopping the camera finishes the sweep, records time to packet and speaks the totals."""
        from library_claim.domain.models import Sweep
        from library_claim.storage.repository import Repository
        count, database = 0, None

        async def receive(session, sent):
            """Start and stop the camera with an object on record, then wait for the summary."""
            nonlocal count, database
            count += 1
            if count == 1:
                database = s.CAPTURE_DATA_DIR / 'claims.sqlite3'
                session.country = 'US'
                return {'type': 'camera_state', 'enabled': True}
            if count == 2:
                # The finish step reloads books from the (empty) database; objects stay in memory.
                session.item_inventory = [{'id': 'lamp'}]
                return {'type': 'camera_state', 'enabled': False}
            while not any('sweep is complete' in t['text'] for t in session.transcript):
                await asyncio.sleep(.001)
            return {'type': 'close'}

        session, _ = await LiveTurnTests.drive(self, receive, library=True)
        with Repository(database) as repo:
            sweep = repo.get(Sweep, 'transport')
        self.assertIsNotNone(sweep.ended_at)
        self.assertIsNotNone(sweep.packet_ready_at)
