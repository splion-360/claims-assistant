"""Text/camera transport stays responsive while inference is in flight."""
import asyncio
import base64
import json
import unittest
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, patch
from live_demo import server as s


class LiveTurnTests(unittest.IsolatedAsyncioTestCase):
    async def drive(self, receive, *, workflow=None, reply=None):
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
        with patch.object(s, '_has_api_key', return_value=True), \
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
        count=0
        async def receive(session,sent):
            nonlocal count
            count+=1
            if count==1:return {'type':'video','data':base64.b64encode(b'\xff\xd8\xffTEST').decode()}
            if count==2:
                while not any(m['type']=='error' for m in sent):await asyncio.sleep(.001)
                clock[0]+=9
                return {'type':'video','data':base64.b64encode(b'\xff\xd8\xffTEST').decode()}
            while len([m for m in sent if m['type']=='error'])<2:await asyncio.sleep(.001)
            return {'type':'close'}
        clock=[100.0]
        # Override only server timing; do not change the asyncio event loop clock.
        with patch.object(s,'time',NS(monotonic=lambda:clock[0])),patch.object(s,'_pin_evidence_photo',side_effect=s.InferenceError('Temporary failure')) as capture:
            await self.drive(receive)
        self.assertEqual(capture.call_count,2)
