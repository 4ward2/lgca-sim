import unittest
from sim.a4.attempts import resolve_attempts
from sim.a4.protocol import digest

class Attempts(unittest.TestCase):
    def test_documented_retry(self):
        first=dict(attempt=0,identity={'seed':173101},status='infrastructure_failure',reason='runner disconnected')
        second=dict(attempt=1,identity={'seed':173101},status='complete',parent_attempt_sha256=digest(first))
        self.assertEqual(resolve_attempts([second,first]),second)

    def test_algorithm_failure_cannot_retry(self):
        first=dict(attempt=0,identity={'seed':173101},status='budget_exceeded')
        second=dict(attempt=1,identity={'seed':173101},status='complete',parent_attempt_sha256=digest(first))
        with self.assertRaises(ValueError): resolve_attempts([first,second])

    def test_duplicate_and_missing_chain(self):
        with self.assertRaises(ValueError): resolve_attempts([])
        first=dict(attempt=0,identity={'seed':173101},status='complete')
        with self.assertRaises(ValueError): resolve_attempts([first,first])
