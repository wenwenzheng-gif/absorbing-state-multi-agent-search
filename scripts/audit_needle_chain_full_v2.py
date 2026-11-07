"""Independent final audit. No provider access and no experiment execution."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.agent_system.no_transition.config import load_config, fingerprint
from src.agent_system.no_transition.full_analysis import iter_episodes
from src.agent_system.no_transition.preservation import verify_full_design, verify_protected_pilots
from src.agent_system.no_transition.runner import episode_id
from src.agent_system.no_transition.report import write_json


def audit():
    config=load_config(ROOT/'configs/needle_chain/full_v2.json')
    folder=ROOT/'results/needle_chain/full_v2'
    raw=ROOT/'runs/needle_chain/full_v2'
    sha='c9246bf47a2f107e79413be6e9f4bda3e3ae9f78'
    expected={(c['name'],kind,s,d) for c in config['conditions'] for kind in ['deterministic','llm']
              for s in config['deterministic_seeds' if kind=='deterministic' else 'llm_seeds'] for d in config['degrees']}
    seen=set();ids=set();cells=Counter();paired=defaultdict(set);behavior=[];technical=[]
    for e in iter_episodes(folder):
        key=e['condition'],e['agent_type'],e['task_seed'],e['degree']
        assert key in expected and key not in seen
        seen.add(key)
        condition=next(c for c in config['conditions'] if c['name']==e['condition'])
        assert e['episode_id']==episode_id(condition,e['agent_type'],e['task_seed'],e['degree'])
        assert e['episode_id'] not in ids
        ids.add(e['episode_id']);cells[e['condition'],e['agent_type'],e['degree']]+=1
        assert e['git_commit_sha']==sha and e['identity']['config_digest']==fingerprint(config)
        assert e['decisions']==20*e['rounds_used']
        for a,neighbors in enumerate(e['graph']):
            assert a not in neighbors and len(neighbors)==len(set(neighbors))==e['degree']
            assert all(a in e['graph'][b] for b in neighbors)
        paired[e['condition'],e['agent_type'],e['task_seed']].add((e['hidden_target'],e['hidden_chain_fingerprint'],e['graph_seed'],e['budget']))
        if e['task']=='chain':
            assert e['success']==(e['maximum_depth']==8)
            assert all(0<=b-a<=1 for a,b in zip([0]+e['depth_after_each_round'],e['depth_after_each_round']))
            if e['success']:
                assert e['round_of_success']>=8 and e['round_of_success']<=e['horizon']
            if e['horizon']==6:
                assert e['maximum_depth']<=6 and not e['success']
        else:
            actual=[a['candidate_tested'] for a in e['actions'] if not a['invalid']]
            assert e['success']==(e['hidden_target'] in actual)
            assert e['total_tests']==len(actual)
            assert e['unique_candidates_tested']==len(set(actual))
            assert e['duplicate_tests']==len(actual)-len(set(actual))
        if e['agent_type']=='llm':
            assert e['model']=='gpt-5.4-nano-2026-03-17'
            assert set(e['response_models'])<={'gpt-5.4-nano-2026-03-17'}
            if e['invalid_actions']:
                requests_path=raw/'episodes'/e['episode_id']/'requests.jsonl'
                requests=[json.loads(line) for line in requests_path.read_text().splitlines()] if requests_path.exists() else []
                for a in e['actions']:
                    if not a['invalid']:
                        continue
                    item={'episode_id':e['episode_id'],'round':a['round'],'agent_id':a['agent_id'],
                          'response_status':a.get('response',{}).get('status',a.get('candidate_result')),
                          'future_attempt':a.get('future_attempt',False),'episode_success':e['success'],
                          'maximum_depth':e['maximum_depth']}
                    matching=[r for r in requests if r['round_number']==a['round'] and r['agent_index']==a['agent_id']]
                    if matching and e['task']=='chain':
                        view=json.loads(matching[0]['messages'][1]['content'])
                        pointer=view['available_pointer']
                        item.update(available_pointer_status=pointer['status'],
                            submitted_current_level=a['level']==pointer.get('level'),
                            submitted_key_matches_available_key=a['key']==pointer.get('key'))
                    behavior.append(item)
            if e['retry_count'] or e['api_errors']:
                technical.append({k:e[k] for k in ['episode_id','api_attempts','successful_api_calls',
                    'api_failed_attempts','retry_count','api_errors','ambiguous_or_transport_interrupted_attempts']})
        else:
            assert e['api_attempts']==0
    assert seen==expected and len(ids)==5088 and len(cells)==48
    assert all(n==(200 if key[1]=='deterministic' else 12) for key,n in cells.items())
    assert all(len(values)==1 for values in paired.values())
    protection=verify_protected_pilots();design=verify_full_design()
    raw_manifest=ROOT/'runs/needle_chain/pilot_raw_before_full_v2_hashes.json'
    pilot_raw={'status':'unavailable_in_this_checkout'}
    if raw_manifest.exists():
        mapping=json.loads(raw_manifest.read_text())
        actual={str(p.relative_to(ROOT)) for label in ['pilot','pilot_v2'] for p in (ROOT/f'runs/needle_chain/{label}').rglob('*') if p.is_file()}
        assert actual==set(mapping)
        assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==digest for p,digest in mapping.items())
        pilot_raw={'status':'unchanged','files':len(mapping)}
    reference=ROOT/'runs/needle_chain/full_v2_reference_import/runs/needle_chain/full_v2/episodes'
    stable_reference={'status':'unavailable_in_this_checkout'}
    if reference.exists() and raw.exists():
        files=list(reference.glob('*/episode_result.json'))
        assert len(files)==4800
        assert all(p.read_bytes()==(raw/'episodes'/p.parent.name/p.name).read_bytes() for p in files)
        stable_reference={'status':'unchanged_after_gpt_stages','episodes':len(files)}
    evidence={'status':'passed','episodes':len(ids),'complete_cells':len(cells),'paired_tasks':len(paired),
        'source_sha':sha,'protected_processed':protection,'protected_local_raw':pilot_raw,
        'frozen_design':design,'reference_checkpoint_preserved':stable_reference,
        'retained_invalid_actions':behavior,'technical_retry_or_error_episodes':technical,
        'checks':['exact episode IDs and fixed seed sets','per-cell sample counts','paired task and graph seed identities',
                  'undirected degree-correct graphs without self edges','actual Needle target tests and duplicate accounting',
                  'actual Chain completion depth and one-level barriers','exact GPT model','no deterministic API calls']}
    write_json(folder/'independent_final_audit.json',evidence)
    print(json.dumps(evidence,indent=2))
    return evidence


if __name__=='__main__':
    audit()
