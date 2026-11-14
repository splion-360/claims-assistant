"""Verify the Drex guidance choice, cooldowns, missed-area count and speaking rule."""
import json
import os
import unittest
from unittest.mock import patch

import httpx

from live_demo.guidance_decision import (LABELS, QUALITY_PROMPTS, GuidanceSignals, QualityPromptLimiter,
                                         choose_prompt, edge_cut_count, should_speak)

SIGNALS = GuidanceSignals(recent_issues=[('glare', 1.0)], gate_failure_streak=0, missing_title_books=0,
                          edge_cut_books=0, last_prompt=None, seconds_since_prompt=None, phase='Sweeping')


def drex_client(handler) -> httpx.Client:
    """Route Drex requests to a test handler.

    Args:
        handler: Callable returning a response for a request.

    Returns:
        An httpx client with a mock transport.
    """
    return httpx.Client(transport=httpx.MockTransport(handler))


class GuidanceDecisionTests(unittest.TestCase):
    """Exercise D6 mapping and the code-side limits that always win."""

    def setUp(self):
        """Provide a Drex key."""
        patcher = patch.dict(os.environ, {'DREX_API_KEY': 'k'})
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_choose_prompt_maps_choice(self):
        """Ask one choice question over every label and return Drex's pick."""
        def handler(request):
            body = json.loads(request.content)
            self.assertEqual(set(body['questions']['prompt']['criteria']), set(LABELS))
            self.assertEqual(body['state']['recent_issues'], [['glare', 1.0]])
            return httpx.Response(200, json={'model': 'drex-max-v1.1', 'answers': {'prompt': {
                'type': 'choice', 'choice': 'reduce_glare', 'probabilities': {'reduce_glare': 0.9}}}})

        self.assertEqual(choose_prompt(SIGNALS, client=drex_client(handler)), 'reduce_glare')

    def test_drex_failure_is_quiet(self):
        """Stay quiet when no decision is available."""
        self.assertEqual(choose_prompt(SIGNALS, client=drex_client(lambda request: httpx.Response(500))), 'stay_quiet')

    def test_limiter_kind_and_label_cooldowns(self):
        """Space quality prompts by 8 s and repeat a label only after 20 s."""
        limiter = QualityPromptLimiter()
        self.assertTrue(limiter.allow('slow_down', 0))
        limiter.record('slow_down', 0)
        self.assertFalse(limiter.allow('move_closer', 7))
        self.assertTrue(limiter.allow('move_closer', 9))
        self.assertFalse(limiter.allow('slow_down', 19))
        self.assertTrue(limiter.allow('slow_down', 21))

    def test_edge_cut_count(self):
        """Count only books never seen without a cut-off edge."""
        inventory = [{'observations': [{'issues': ['cut_off']}, {'issues': ['cut_off', 'blur']}]},
                     {'observations': [{'issues': ['cut_off']}, {'issues': []}]},
                     {'observations': []}]
        self.assertEqual(edge_cut_count(inventory), 1)

    def test_no_prompt_while_user_speaking(self):
        """Never talk over the user or over the agent's own speech."""
        self.assertFalse(should_speak(user_speaking=True, agent_speaking=False))
        self.assertFalse(should_speak(user_speaking=False, agent_speaking=True))
        self.assertTrue(should_speak(user_speaking=False, agent_speaking=False))

    def test_quality_prompt_copy(self):
        """Keep the fixed spoken sentences from the plan."""
        self.assertEqual(QUALITY_PROMPTS['reduce_glare'], 'Shift the camera slightly to the side to avoid the glare.')
        self.assertEqual(LABELS[0], 'stay_quiet')
        self.assertEqual(LABELS[-1], 'read_title')
