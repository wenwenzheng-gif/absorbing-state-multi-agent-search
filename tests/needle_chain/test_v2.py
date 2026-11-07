import asyncio
import copy
import hashlib
import json

import pytest

from src.agent_system.no_transition.analysis import aggregate, verify_invariants
from src.agent_system.no_transition.config import (ROOT, artifact_label, load_config,
    expected_episode_count, request_bound, validate)
from src.agent_system.no_transition.environments import (NeedleEnvironment, candidate_order,
    public_schedule, reference_batch)
from src.agent_system.no_transition.policies import (NEEDLE_PROMPT, CHAIN_PROMPT,
    NEEDLE_HIGH_V2_PROMPT, ReferencePolicy, messages_and_schema)
from src.agent_system.no_transition.preservation import guard_v2_paths, verify_frozen_v1
from src.agent_system.no_transition.report_v2 import analyze_v2, coverage_records, readiness, observed_findings
from src.agent_system.no_transition.runner import run_episode


def config():
    return load_config(ROOT / 'configs/needle_chain/pilot_v2.json')


def test_frozen_v1_hashes_and_unmodified_conditions_and_prompts():
    assert verify_frozen_v1()['verified_files'] == 22
    old = load_config(ROOT / 'configs/needle_chain/pilot.json')
    new = config()
    assert all(new['conditions'][i] == old['conditions'][i] for i in [0, 2, 3])
    assert new['llm'] == old['llm']
    # Historical constant hashes also work in a shallow hosted checkout.
    hashes = json.loads((ROOT / 'docs/needle_chain/v1_frozen_hashes.json').read_text())['prompt_sha256']
    assert hashlib.sha256(NEEDLE_PROMPT.encode()).hexdigest() == hashes['NEEDLE_PROMPT']
    assert hashlib.sha256(CHAIN_PROMPT.encode()).hexdigest() == hashes['CHAIN_PROMPT']


def test_v2_sizes_and_paths_cannot_overwrite_v1():
    c = config()
    assert request_bound(c) == 864
    assert expected_episode_count(c, 'llm') == 36
    assert expected_episode_count(c) == 516
    assert artifact_label(c) == 'pilot_v2'
    for name in ['runs', 'results']:
        with pytest.raises(ValueError, match='frozen v1'):
            guard_v2_paths(ROOT / f'{name}/needle_chain/pilot/subdir', ROOT / 'results/needle_chain/pilot_v2')
    c['mode'] = 'full'
    with pytest.raises(AssertionError, match='pilot-only'):
        validate(c)


def test_two_complete_schedule_passes_use_no_target_or_degree(monkeypatch):
    c = config()['conditions'][1]
    for seed in range(91000, 91040):
        schedule = public_schedule(c, seed)
        assert len(schedule) == 96
        assert schedule[:48] == candidate_order(48, seed)
        assert set(schedule[:48]) == set(schedule[48:]) == set(range(48))
        assert schedule[:48] != schedule[48:]
        assignments = [x for r in [1, 2] for a in range(6)
                       for x in reference_batch(schedule, 6, a, r, 8)]
        assert assignments == schedule
    env = NeedleEnvironment(48, 91000)
    before = public_schedule(c, 91000)
    for target in range(48):
        env._target = target
        assert public_schedule(c, 91000) == before
        assert before.count(target) == 2


def test_slack_reference_actions_and_prompt_views_match_across_degrees(tmp_path):
    c, views, results = config(), {}, []
    class Capture(ReferencePolicy):
        async def decide(self, view, context):
            views.setdefault((context.round_number, context.agent_index), []).append(copy.deepcopy(view))
            return await super().decide(view, context)
    for degree in c['degrees']:
        results.append(asyncio.run(run_episode(c, c['conditions'][1], 'deterministic', 91000,
                                               degree, tmp_path, policy_override=Capture())))
    for e in results:
        assert e['success'] and e['total_tests'] == 96 and e['unique_candidates_tested'] == 48
        assert e['duplicate_tests'] == 48 and e['round_of_success'] == 1
        assert e['api_attempts'] == 0
    assert verify_invariants(results, c)['status'] == 'passed'
    for group in views.values():
        stripped = [{k: v for k, v in view.items() if k != 'neighbor_messages'} for view in group]
        assert stripped[0] == stripped[1] == stripped[2]
    detail = coverage_records(results, 'v2')
    assert all(e['scheduled_candidates_followed'] == 96 and e['scheduled_candidates_replaced'] == 0
               and e['nominal_target_occurrences'] == 2 for e in detail)


def test_guidance_is_high_only_and_schema_preserves_real_choice():
    view = {'task': 'needle', 'batch_size': 8, 'policy_variant': 'needle_high_v2'}
    messages, schema = messages_and_schema(view)
    assert messages[0]['content'] == NEEDLE_HIGH_V2_PROMPT
    assert 'phase transition' not in messages[0]['content']
    assert schema['properties']['candidates']['items'] == {'type': 'integer'}
    assert 'enum' not in json.dumps(schema) and 'const' not in json.dumps(schema)
    assert messages_and_schema({'task': 'needle', 'batch_size': 4})[0][0]['content'] == NEEDLE_PROMPT


def test_departures_count_multiset_matches_not_order_and_failures_survive(tmp_path):
    c = config()
    class Reverse(ReferencePolicy):
        async def decide(self, view, context):
            return {'action': {'candidates': list(reversed(view['scheduled_candidates']))},
                    'api_error': False, 'behavior_error': False}
    out = asyncio.run(run_episode(c, c['conditions'][1], 'llm', 91000, 0, tmp_path/'reverse', policy_override=Reverse()))
    assert coverage_records([out], 'v2')[0]['schedule_departure_fraction'] == 0
    class Duplicates(ReferencePolicy):
        async def decide(self, view, context):
            return {'action': {'candidates': [0]*8}, 'api_error': False, 'behavior_error': False}
    out = asyncio.run(run_episode(c, c['conditions'][1], 'llm', 91000, 0, tmp_path/'duplicate', policy_override=Duplicates()))
    record = coverage_records([out], 'v2')[0]
    assert not record['success'] and record['total_tests'] == 96 and record['unique_candidates_tested'] == 1
    assert record['scheduled_candidates_followed'] == 2 and record['scheduled_candidates_replaced'] == 94
    assert record['target_in_nominal_schedule'] and not record['target_actually_tested']
    findings = observed_findings([out])
    assert findings['first_round_full_coverage'] == 0 and findings['successes'] == 0
    assert findings['changed_batches'] == 12 and findings['observed_second_pass_target_rescues'] == 0
    assert readiness(aggregate([out], c), c, False, 'passed')['status'] == 'NOT READY'


def test_readiness_does_not_treat_tiny_pilot_as_proof_or_authorization():
    c = config()
    rows = [{'condition': condition['name'], 'agent_type': kind, 'degree': d,
             'p_success': .025 if condition['name'] == 'Needle-Low' else
                          0 if condition['name'] == 'Chain-Short-Horizon' else 1,
             'ci_low': 0, 'ci_high': 1, 'coverage_fraction': 1}
            for condition in c['conditions'] for kind in ['deterministic', 'llm'] for d in c['degrees']]
    result = readiness(rows, c, True, 'passed')
    assert result['status'] == 'READY FOR FULL RUN'
    assert not result['absence_of_transition_established'] and not result['full_run_authorized']
    high = next(r for r in rows if r['condition'] == 'Needle-High' and r['agent_type'] == 'llm')
    high['p_success'] = 4/6
    assert readiness(rows, c, True, 'passed')['status'] == 'INCONCLUSIVE'
    high['coverage_fraction'] = .8
    assert readiness(rows, c, True, 'passed')['status'] == 'NOT READY'


def test_v2_offline_end_to_end_keeps_versions_separate(tmp_path):
    c = config()
    c['deterministic_seeds'] = c['llm_seeds'] = [91000]
    c['llm_seeds_by_condition'] = {'Needle-High': [91000]}
    run, output = tmp_path/'run', tmp_path/'output'
    for condition in c['conditions']:
        for kind in ['deterministic', 'llm']:
            for degree in c['degrees']:
                asyncio.run(run_episode(c, condition, kind, 91000, degree, run,
                                        policy_override=ReferencePolicy()))
    summary = analyze_v2(run, output, c, {})
    assert summary['complete'] and summary['episodes'] == 24
    assert len(list((output/'figures').glob('*'))) == 8
    assert (output/'pilot_report_v2.md').exists() and not (output/'pilot_report.md').exists()
    records = json.loads((output/'needle_high_behavior.json').read_text())
    assert len(records) == 6 and all(r['version'] == 'v2' for r in records)
    assert verify_frozen_v1()['status'] == 'unchanged'
