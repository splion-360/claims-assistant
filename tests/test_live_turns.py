"""Text/camera transport stays responsive while inference is in flight."""
import asyncio
import base64
import json
import tempfile
from pathlib import Path
import unittest
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, patch
from live_demo import server as s


class LiveTurnTests(unittest.IsolatedAsyncioTestCase):
    async def drive(self, receive, *, workflow=None, reply=None, relevance=None):
        """Exercise live transport with isolated evidence storage.

        Args:
            receive: Async function supplying simulated browser messages.
            workflow: Optional background workflow implementation.
            reply: Optional conversation implementation.
            relevance: Optional local relevance test double.

        Returns:
            The session and emitted transport messages.
        """
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        session = s.IntakeSession('transport', owner='owner')
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
        with patch.object(s, 'CameraRelevance', return_value=relevance), \
             patch.object(s, 'CAPTURE_DATA_DIR', Path(directory.name)), \
             patch.object(s, '_has_api_key', return_value=True), \
             patch.object(s, 'run_claim_workflow', side_effect=workflow or AsyncMock(return_value=s.build_initial_workflow_state())), \
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

    async def test_slow_workflow_does_not_block_camera_or_close(self):
        started=asyncio.Event();count=0
        async def workflow(*args,**kwargs):
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
        session,_=await self.drive(receive,workflow=workflow)
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
        from library_claim.capture.relevance.classifier import decide

        _, image = cv2.imencode('.jpg', textured_view())
        jpeg = image.tobytes()
        relevance = Mock(assess=Mock(return_value=decide({'books': .4}, 1)))
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
            await self.drive(receive, relevance=relevance)
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

    async def test_reply_does_not_wait_for_background_packet(self):
        count=0
        async def workflow(*args,**kwargs):await asyncio.Event().wait()
        async def receive(session,sent):
            nonlocal count
            count+=1
            if count==1:return {'type':'text','text':'Hello','id':'one'}
            while not any(m['type']=='transcript' and m['speaker']=='Agent' for m in sent):await asyncio.sleep(.001)
            return {'type':'close'}
        session,_=await self.drive(receive,workflow=workflow)
        self.assertEqual(session.transcript[-1]['text'],'What happened?')
        self.assertFalse(session.tasks)
