"""Verify the sweep stays local-only until a public host is named, then admits just that host."""
import os
import unittest
from unittest.mock import patch

import live_demo.server as s


class PublicHostTests(unittest.TestCase):
    """Exercise the host and origin checks a tunnel has to pass."""

    @staticmethod
    def without_public_host():
        """Clear the setting so the sweep is local-only.

        Returns:
            A patch context removing the public host.
        """
        return patch.dict(os.environ, {'PUBLIC_HOST': '', 'FNOL_CORS_ORIGINS': ''})

    @staticmethod
    def with_public_host(value='demo-123.ngrok-free.app'):
        """Name a tunnel host the sweep may also be served on.

        Args:
            value: The configured host.

        Returns:
            A patch context setting the public host.
        """
        return patch.dict(os.environ, {'PUBLIC_HOST': value, 'FNOL_CORS_ORIGINS': ''})

    def test_local_hosts_are_always_allowed(self):
        """Serve localhost whether or not a tunnel is configured."""
        with self.without_public_host():
            self.assertTrue(s.allowed_host('127.0.0.1:4177'))
            self.assertTrue(s.allowed_host('localhost:4177'))

    def test_without_the_setting_no_other_host_is_served(self):
        """Refuse a tunnel host until it is named in the environment."""
        with self.without_public_host():
            self.assertFalse(s.allowed_host('demo-123.ngrok-free.app'))
            self.assertEqual(s.public_host(), '')
            self.assertNotIn('https://demo-123.ngrok-free.app', s._cors_origins())

    def test_the_named_host_is_served_over_either_scheme(self):
        """Admit the configured host, which a tunnel reaches over https."""
        with self.with_public_host():
            self.assertTrue(s.allowed_host('demo-123.ngrok-free.app'))
            origins = s._cors_origins()
            self.assertIn('https://demo-123.ngrok-free.app', origins)
            self.assertIn('http://127.0.0.1:4177', origins)

    def test_another_host_is_still_refused(self):
        """Admit only the named host, not any other public name."""
        with self.with_public_host():
            self.assertFalse(s.allowed_host('attacker.example.com'))

    def test_a_url_shaped_setting_is_reduced_to_its_host(self):
        """Accept the setting written as a full URL, as a tunnel prints it."""
        with self.with_public_host('https://demo-123.ngrok-free.app/'):
            self.assertEqual(s.public_host(), 'demo-123.ngrok-free.app')
            self.assertTrue(s.allowed_host('demo-123.ngrok-free.app'))

    def test_the_tunnel_origin_passes_the_origin_check(self):
        """Accept the browser's https origin although this process sees http."""
        with self.with_public_host():
            self.assertTrue(s.allowed_origin('https://demo-123.ngrok-free.app',
                                             'demo-123.ngrok-free.app', 'http'))

    def test_an_unnamed_origin_is_refused_behind_a_tunnel(self):
        """Refuse an origin that is neither local nor the configured host."""
        with self.with_public_host():
            self.assertFalse(s.allowed_origin('https://elsewhere.example.com',
                                              'demo-123.ngrok-free.app', 'http'))

    def test_an_explicit_origin_list_still_wins(self):
        """Let an explicit list replace the defaults, as it did before."""
        with patch.dict(os.environ, {'PUBLIC_HOST': 'demo-123.ngrok-free.app',
                                     'FNOL_CORS_ORIGINS': 'http://example.test'}):
            self.assertEqual(s._cors_origins(), ['http://example.test'])

    def test_without_the_setting_only_local_clients_connect(self):
        """Refuse a remote address while the sweep is local-only."""
        with self.without_public_host():
            self.assertTrue(s.allowed_client('127.0.0.1'))
            self.assertTrue(s.allowed_client('::1'))
            self.assertFalse(s.allowed_client('148.64.100.237'))

    def test_naming_a_public_host_admits_remote_clients(self):
        """Stop narrowing the address once the sweep is deliberately exposed.

        A tunnel forwards the original client's address, so remote peers are
        exactly what a configured sweep expects; the host and origin checks are
        what guard it from then on.
        """
        with self.with_public_host():
            self.assertTrue(s.allowed_client('148.64.100.237'))
            self.assertTrue(s.allowed_client('127.0.0.1'))
