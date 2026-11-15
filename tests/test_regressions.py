"""Regression tests for the live session transport, evidence capture and access control."""
import asyncio, base64, time, unittest
from unittest.mock import patch

from live_demo import server as s
from fastapi.testclient import TestClient


class TranscriptTests(unittest.TestCase):
    def test_agent_reply_does_not_bump_the_revision(self):
        session=s.IntakeSession('revision')
        s.append_turn(session,'Claimant','That shelf is not mine','u1')
        revision=session.revision
        s.append_turn(session,'Agent','Noted.','a1')
        self.assertEqual(session.revision,revision)

    def test_turn_retry_deduplicates_only_identifier(self):
        session=s.IntakeSession('test')
        s.append_turn(session,'Claimant','No','1');s.append_turn(session,'Claimant','No','2');s.append_turn(session,'Claimant','No','2')
        self.assertEqual(len(session.transcript),2)


class AsyncTests(unittest.IsolatedAsyncioTestCase):
    async def test_exact_frozen_frame_reverified_despite_new_arrival(self):
        session=s.IntakeSession('frame',last_frame=b'FRAME_A',last_frame_id='a',last_frame_at=time.monotonic())
        async def model(instruction, content, schema, **kwargs):
            self.assertEqual(base64.b64decode(content[1]['image_url']['url'].split(',')[1]),b'FRAME_A')
            session.last_frame=b'FRAME_B'
            return s.FrameObservation(observation='Blank wall',supports_claimant_description=False)
        with patch.object(s,'structured',side_effect=model):
            result=await s._pin_evidence_photo(session,dict(observation='Huge crack',confirmed=True,claimant_description='A crack'))
        photo=session.evidence_photos[0]
        self.assertEqual(base64.b64decode(photo['data_url'].split(',')[1]),b'FRAME_A')
        self.assertEqual(photo['caption'],'Blank wall')
        self.assertFalse(result['confirmed']);self.assertIn('unconfirmed',session.camera_notes[0])

    async def test_without_claimant_description_capture_is_unconfirmed(self):
        session=s.IntakeSession('frame',last_frame=b'FRAME',last_frame_at=time.monotonic())
        async def model(*args, **kwargs): return s.FrameObservation(observation='Wall',supports_claimant_description=True)
        with patch.object(s,'structured',side_effect=model):
            self.assertFalse((await s._pin_evidence_photo(session,{}))['confirmed'])
        self.assertIn('unconfirmed',session.camera_notes[0])

    async def test_stale_frames_and_photo_limits(self):
        session=s.IntakeSession('frame',last_frame=b'OLD',last_frame_at=time.monotonic()-30)
        self.assertFalse((await s._pin_evidence_photo(session,{}))['pinned'])
        session.last_frame_at=time.monotonic();session.read_count=s.CONFIG.capture.max_reads_per_session
        self.assertFalse((await s._pin_evidence_photo(session,{}))['pinned'])


    async def test_camera_off_clears_live_frame_but_retains_pinned_photos(self):
        photo={'id':'pinned'}
        session=s.IntakeSession('camera',camera_enabled=True,last_frame=b'frame',last_frame_id='frame-id',last_frame_at=time.monotonic(),evidence_photos=[photo])
        s.set_camera_mode(session,False)
        self.assertFalse(session.camera_enabled);self.assertIsNone(session.last_frame)
        self.assertEqual(session.evidence_photos,[photo])
        self.assertFalse((await s._pin_evidence_photo(session,{}))['pinned'])



    async def test_cleanup_deletes_frames_and_cancels_jobs(self):
        session=s.IntakeSession('expired',updated_at=time.monotonic()-s.SESSION_TTL-1,last_frame=b'image',evidence_photos=[{}])
        job=asyncio.create_task(asyncio.sleep(100));session.tasks.add(job);s.sessions[session.session_id]=session
        await s.cleanup_sessions()
        self.assertNotIn('expired',s.sessions);self.assertTrue(job.cancelled());self.assertIsNone(session.last_frame);self.assertEqual(session.evidence_photos,[])



class AccessTests(unittest.TestCase):
    def setUp(self):
        # App startup prunes old sweeps; point it at a temporary folder so tests never touch real captures.
        import tempfile
        from pathlib import Path
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        guard = unittest.mock.patch.object(s, 'CAPTURE_DATA_DIR', Path(directory.name))
        guard.start()
        self.addCleanup(guard.stop)
        s.sessions.clear();self.client=TestClient(s.app,base_url='http://127.0.0.1:4188');self.headers={'Origin':'http://127.0.0.1:4188'}
    def tearDown(self):self.client.close();s.sessions.clear()
    def create(self):return self.client.post('/api/sessions',headers=self.headers).json()['session_id']
    def test_owner_cookie_required(self):
        sid=self.create();self.assertEqual(self.client.get('/api/sessions/'+sid).status_code,200)
        with TestClient(s.app,base_url='http://127.0.0.1:4188') as other:
            self.assertEqual(other.get('/api/sessions/'+sid).status_code,404)
    def test_origin_host_and_missing_origin_rejected(self):
        for headers in [{},{'Origin':'https://evil.example'},{'Origin':'http://evil.example','Host':'evil.example'}]:
            self.assertEqual(self.client.post('/api/sessions',headers=headers).status_code,403)
    def test_websocket_rejects_origin_and_wrong_owner(self):
        from starlette.websockets import WebSocketDisconnect
        sid=self.create()
        with self.assertRaises(WebSocketDisconnect):
            with self.client.websocket_connect('/ws/live?session_id='+sid,headers={'Origin':'https://evil.example'}):pass
        self.client.cookies.clear()
        with self.assertRaises(WebSocketDisconnect):
            with self.client.websocket_connect('/ws/live?session_id='+sid,headers=self.headers):pass
    def test_reset_deletes_old_claim(self):
        old=self.create();s.append_turn(s.sessions[old],'Claimant','Old confidential claim')
        self.assertEqual(self.client.delete('/api/sessions/'+old,headers=self.headers).status_code,200)
        new=self.create();self.assertNotEqual(old,new)
        self.assertEqual(self.client.get('/api/sessions/'+old).status_code,404)
        self.assertEqual(len(self.client.get('/api/sessions/'+new).json()['state']['transcript']),1)
    def test_session_state_is_the_library_sweep(self):
        state=self.client.get('/api/sessions/'+self.create()).json()['state']
        self.assertTrue(state['library_sweep'])
        self.assertNotIn('policy',state)
    def test_session_limit_and_source_not_served(self):
        for _ in range(4):self.create()
        self.assertEqual(self.client.post('/api/sessions',headers=self.headers).status_code,429)
        self.assertEqual(self.client.get('/server.py').status_code,404)


if __name__=='__main__':unittest.main(verbosity=2)
