"""Infrastructure resume must preserve ledgers and exact prior model samples."""
import asyncio
import json
from pathlib import Path
import tempfile
import unittest
from src.agent_system.storage import RunStore
from src.agent_system.policies.openai_policy import LLMBackend, Usage

class ResumeTests(unittest.TestCase):
    def test_only_successful_episodes_are_skipped_without_rewriting_ledger(self):
        with tempfile.TemporaryDirectory() as root:
            with RunStore(Path(root), 'seed_0') as store:
                store.append('episodes', {'episode_id':'done','status':'ok'})
                store.append('episodes', {'episode_id':'unfinished','status':'budget_paused'})
                path = store.path / 'episodes.jsonl'
                original = path.read_bytes()
                self.assertEqual(store.completed_episodes(), {'done'})
                store.append('episodes', {'episode_id':'unfinished','status':'ok'})
                self.assertTrue(path.read_bytes().startswith(original))
                self.assertEqual(store.completed_episodes(), {'done','unfinished'})

    def test_recorded_response_returns_without_network_or_ledger_append(self):
        backend = object.__new__(LLMBackend)
        record = {'cache_key':'exact-key','content':'{"choice":1}'}
        backend.resume_responses = {'exact-key':record}
        backend.usage = Usage()
        # No client or sinks: a cache miss would fail, a hit must use neither.
        result = asyncio.run(backend.complete(messages=[],schema={},schema_name='x',cache_key='exact-key',metadata={}))
        self.assertIs(result, record)

if __name__ == '__main__':
    unittest.main()
