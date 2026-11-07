"""Audit saved call identities without emitting credentials or prompt content."""
import gzip
import json
from collections import Counter
from pathlib import Path

from .config import resolved_llm_settings


def audit_calls(run_dir: Path, output: Path, config: dict, episodes: list[dict]) -> dict:
    keys, response_ids, models, sources = set(), set(), Counter(), set()
    request_lines = calls = 0
    episode_keys = set()
    with gzip.open(output/'provider_response_inventory.jsonl.gz', 'wt') as inventory:
        for e in episodes:
            if e['agent_type'] != 'llm':
                continue
            folder = run_dir/'episodes'/e['episode_id']
            if not folder.exists():
                return {'status':'unavailable_offline_archive', 'note':'Use the preserved hosted raw-call audit.'}
            requests = folder/'requests.jsonl'
            local = set()
            if requests.exists():
                for line in requests.read_text().splitlines():
                    r = json.loads(line)
                    request_lines += 1
                    assert r['model'] == resolved_llm_settings(config)['model']
                    local.add(r['cache_key'])
            assert not (keys & local), 'request key shared between episodes'
            keys.update(local)
            episode_keys.add(e['episode_id'])
            for path in sorted((folder/'calls').glob('*.json')):
                r=json.loads(path.read_text())
                assert r['cache_key'] in local
                assert r['response_id'] not in response_ids
                response_ids.add(r['response_id'])
                models[r['model']] += 1
                assert r['model'] == resolved_llm_settings(config)['model']
                calls += 1
                inventory.write(json.dumps({k:r[k] for k in ['cache_key','response_id','model','usage','episode_id','agent_index','round_number','latency_s']},sort_keys=True)+'\n')
            sources.add(e['git_commit_sha'])
    assert calls == sum(e['successful_api_calls'] for e in episodes)
    assert len(sources)==1
    return {'status':'passed','logical_request_keys':len(keys),'request_ledger_lines':request_lines,
            'successful_responses':calls,'unique_response_ids':len(response_ids),
            'response_model_counts':dict(models),'llm_episodes_audited':len(episode_keys),
            'source_shas':sorted(sources), 'cross_episode_key_collisions':0,
            'ambiguous_attempts':sum(e['ambiguous_or_transport_interrupted_attempts'] for e in episodes)}
