"""Verify automatic guidance stays quiet unless a title needs evidence."""
import unittest
from live_demo.book_guidance import GuidancePolicy, needs_capture, capture_instruction
from library_claim.config import CONFIG


class GuidancePolicyTests(unittest.TestCase):
    """Exercise missing-title objectives, cooldown, and fragment suppression."""

    def test_readable_books_do_not_trigger_guidance(self):
        """Keep duplicate association, blur and absent publishers in review."""
        book = {'id':'a', 'title':'Good People', 'publisher':None, 'author':None,
                'status':'unresolved', 'needs_review':True, 'frame_ids':['frame'], 'candidates':['b'],
                'observations':[{'issues':['blur']}]}
        self.assertFalse(needs_capture(book))
        self.assertIsNone(GuidancePolicy().choose([book], 10, 'frame'))

    def test_unknown_fragments_do_not_repeat_requests(self):
        """Avoid treating unidentifiable new record IDs as new prompts."""
        policy = GuidancePolicy()
        book = {'id':'a', 'title':None, 'frame_ids':['frame']}
        self.assertIsNotNone(policy.choose([book], 0, 'frame'))
        policy.remember(book, 0)
        later = CONFIG.identification.guidance_cooldown_s + 1
        self.assertIsNone(policy.choose([{'id':'fragment', 'frame_ids':['frame']}], later, 'frame'))

    def test_alias_memory_and_cooldown_survive_reconciliation(self):
        """Remember a requested sighting after its canonical identity changes."""
        policy = GuidancePolicy()
        policy.remember({'id':'old', 'title':'Good People'}, 10)
        merged = {'id':'new', 'member_ids':['old','new'], 'title':None, 'frame_ids':['frame']}
        self.assertIsNone(policy.choose([merged], 10 + CONFIG.identification.guidance_cooldown_s + 1, 'frame'))
        new = {'id':'other', 'title':None, 'frame_ids':['frame']}
        self.assertIsNone(policy.choose([new], 11, 'frame'))
        self.assertIsNone(policy.choose([new], 100, 'offscreen'))

    def test_camera_instructions_require_no_capture_controls(self):
        """Request camera adjustments instead of moving books or pressing buttons."""
        for issue in ['blur','glare','occlusion','cut_off']:
            text = capture_instruction({'observations':[{'issues':[issue]}]}).lower()
            self.assertNotIn('tap', text)
            self.assertNotIn('ready', text)
            self.assertNotIn('hold the book', text)

    def test_excluded_books_never_need_capture(self):
        """Books the user skipped never trigger a title request."""
        self.assertFalse(needs_capture({'id': 'a', 'title': None, 'excluded': True}))
        self.assertTrue(needs_capture({'id': 'a', 'title': None, 'excluded': False}))
