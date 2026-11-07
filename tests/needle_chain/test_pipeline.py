import asyncio
import copy
import json
from pathlib import Path

import httpx
import numpy as np
import pytest

from src.agent_system.no_transition.analysis import bh_adjust, degree_effect, logistic_fit, paired_endpoint, wilson
from src.agent_system.no_transition.config import ROOT, fingerprint, load_config, request_bound, threshold, resolved_llm_settings
from src.agent_system.no_transition.policies import NanoPolicy, ReferencePolicy, messages_and_schema
from src.agent_system.no_transition.runner import run_episode
from src.agent_system.policies.base import DecisionContext
from src.agent_system.policies.openai_policy import LLMBackend
from src.agent_system.schemas import LLMConfig
from src.agent_system.storage import RunStore


def config():
    return load_config(ROOT / 'configs/needle_chain/pilot.json')


def test_pilot_and_full_configuration_bounds():
    c = config()
    assert request_bound(c) == 720
    full = load_config(ROOT / 'configs/needle_chain/full.json')
    assert full['agents'] == 20 and len(full['deterministic_seeds']) >= 100
    assert request_bound(full) == 14400
    assert full['degrees'] == [0, 4, 8, 12, 16, 19]


def test_exact_snapshot_is_used_without_mutating_frozen_resources():
    c = config()
    original = copy.deepcopy(c)
    settings = resolved_llm_settings(c)
    assert settings['model'] == 'gpt-5.4-nano-2026-03-17'
    assert c == original
    with pytest.raises(ValueError, match='snapshot'):
        resolved_llm_settings(c, {'OPENAI_MODEL': 'gpt-5.4-nano'})


def test_hosted_restore_reuses_committed_results_and_is_idempotent(tmp_path):
    from src.agent_system.no_transition.hosted import restore_completed
    source = ROOT / 'results/needle_chain/pilot'
    c = config()
    count = restore_completed(source, tmp_path, c)
    assert count >= 480
    assert restore_completed(source, tmp_path, c) == 0
    assert len(list((tmp_path / 'episodes').glob('*/episode_result.json'))) == count
    full = load_config(ROOT / 'configs/needle_chain/full.json')
    with pytest.raises(ValueError, match='pilot-only'):
        restore_completed(source, tmp_path, full)


@pytest.mark.parametrize('k,n', [(0, 2), (2, 2), (0, 40), (40, 40), (1, 40)])
def test_wilson_boundary_counts(k, n):
    lo, hi = wilson(k, n)
    assert 0 <= lo <= k / n + 1e-14 <= hi + 1e-14 <= 1 + 1e-14
    assert hi > lo
    json.dumps([lo, hi], allow_nan=False)


def test_theory_labels_are_json_safe_and_never_fitted():
    c = config()
    assert [threshold(x)['dc'] for x in c['conditions']] == [0, 0, 'infinity', 0]
    json.dumps([threshold(x) for x in c['conditions']], allow_nan=False)
    # The new analysis may reuse CSV utilities, but never the old dc estimators.
    for name in ['analysis.py', 'report.py', 'plots.py', 'runner.py']:
        source = (ROOT / 'src/agent_system/no_transition' / name).read_text()
        assert 'import critical_degree' not in source
        assert 'from .critical_degree' not in source
        assert 'import success_crossover' not in source


def test_logistic_constants_separation_and_identifiable_degree_effect():
    x = np.tile([0., 2., 5.], 100)
    for outcome in (0, 1):
        assert logistic_fit(x, np.full(len(x), outcome))['status'] == 'not_identifiable_constant_outcome'
    assert logistic_fit(np.array([0., 0., 5., 5.]), np.array([0., 0., 1., 1.]))['status'] == 'not_identifiable_separation'
    rng = np.random.default_rng(14)
    y = rng.random(len(x)) < 1 / (1 + np.exp(-(-1 + .5 * x)))
    fit = logistic_fit(x, y.astype(float))
    assert fit['status'] == 'estimated' and .1 < fit['coefficient'] < 1
    assert fit['log_likelihood_degree'] > fit['log_likelihood_constant']


def test_paired_endpoint_ci_is_not_falsely_certain_at_boundaries():
    es = [{'task_seed': s, 'degree': d, 'success': True} for s in [1, 2] for d in [0, 2, 5]]
    out = paired_endpoint(es, [0, 2, 5], np.random.default_rng(4), 100)
    assert out['delta_p'] == 0 and out['bootstrap_degenerate']
    assert out['ci95_conservative_wilson'][0] < -.5
    assert out['ci95_conservative_wilson'][1] > .5
    assert out['p_value'] == 1
    assert degree_effect(es, np.random.default_rng(4), 100)['coefficient'] is None


def test_bh_multiple_testing_and_missing_values():
    assert bh_adjust([.01, .04, None, .03]) == pytest.approx([.03, .04, None, .04], nan_ok=True)


def test_identical_paired_outcomes_do_not_get_spurious_wald_precision():
    es = [{'task_seed': s, 'degree': d, 'success': s == 0}
          for s in range(40) for d in [0, 2, 5]]
    result = degree_effect(es, np.random.default_rng(3), 100)
    assert result['bootstrap_degenerate'] and result['p_value'] is None
    assert abs(result['coefficient']) < 1e-12


def test_deterministic_episode_metrics_and_degree_pairing(tmp_path):
    c = config()
    for condition in c['conditions']:
        results = [asyncio.run(run_episode(c, condition, 'deterministic', 11, d, tmp_path)) for d in c['degrees']]
        assert len({r['success'] for r in results}) == 1
        for r in results:
            assert r['invalid_actions'] == 0 and r['api_attempts'] == 0
            assert all(len(v) == r['degree'] for v in r['graph'])
            if condition['task'] == 'needle':
                assert r['unique_candidates_tested'] == 48 and r['duplicate_tests'] == 0
            else:
                assert r['maximum_depth'] == min(condition['chain_length'], condition['horizon'])


def test_completed_episode_never_rerun_and_config_change_rejected(tmp_path):
    c, policy = config(), ReferencePolicy()
    condition = c['conditions'][1]
    first = asyncio.run(run_episode(c, condition, 'deterministic', 10, 0, tmp_path))
    class CannotRun:
        async def decide(self, *args):
            raise AssertionError('completed episode was repeated')
    second = asyncio.run(run_episode(c, condition, 'deterministic', 10, 0, tmp_path, policy_override=CannotRun()))
    assert first['actions'] == second['actions']
    changed = copy.deepcopy(c)
    changed['llm']['temperature'] = .5
    with pytest.raises(ValueError, match='configuration differs'):
        asyncio.run(run_episode(changed, condition, 'deterministic', 10, 0, tmp_path))


def test_interrupted_episode_resumes_from_global_barrier(tmp_path):
    c, seen = config(), []
    class Interrupt(ReferencePolicy):
        async def decide(self, view, context):
            if context.round_number == 2:
                raise RuntimeError('simulated interruption')
            return await super().decide(view, context)
    class Resume(ReferencePolicy):
        async def decide(self, view, context):
            seen.append(context.round_number)
            return await super().decide(view, context)
    condition = c['conditions'][3]
    with pytest.raises(RuntimeError, match='simulated interruption'):
        asyncio.run(run_episode(c, condition, 'deterministic', 2, 2, tmp_path, policy_override=Interrupt()))
    restored = asyncio.run(run_episode(c, condition, 'deterministic', 2, 2, tmp_path, policy_override=Resume()))
    fresh = asyncio.run(run_episode(c, condition, 'deterministic', 2, 2, tmp_path / 'fresh'))
    assert min(seen) == 2
    assert restored['actions'] == fresh['actions'] and restored['messages'] == fresh['messages']
    assert restored['depth_after_each_round'] == list(range(1, 9))


def test_no_hidden_state_or_degree_in_prompts(tmp_path):
    c, views = config(), []
    class Inspect(ReferencePolicy):
        async def decide(self, view, context):
            # Freeze before runner history is updated.
            views.append(json.loads(json.dumps(view)))
            return await super().decide(view, context)
    for d in [0, 5]:
        asyncio.run(run_episode(c, c['conditions'][1], 'deterministic', 8, d, tmp_path, policy_override=Inspect()))
    for view in views:
        messages, schema = messages_and_schema(view)
        assert not any(key in view for key in ['hidden_target', 'task_seed', 'seed', 'degree', 'graph_seed'])
        assert 'phase transition' not in str(messages).lower()
        assert 'dc' not in view
    assert views[:6] == views[12:18]  # first-round inputs identical across degrees
    assert not views[6]['neighbor_messages'] and views[18]['neighbor_messages']


def test_invalid_llm_decisions_are_retained_without_reference_fallback(tmp_path):
    c = config()
    class Invalid:
        async def decide(self, view, context):
            return {'action': {'candidates': [-1] * 4}, 'api_error': False, 'behavior_error': False}
    out = asyncio.run(run_episode(c, c['conditions'][1], 'llm', 8, 0, tmp_path, policy_override=Invalid()))
    assert not out['success'] and out['invalid_actions'] == 48 and out['unique_candidates_tested'] == 0


def test_durable_backend_response_replayed_after_interrupt(tmp_path):
    async def check():
        cfg = LLMConfig(**config()['llm'])
        with RunStore(tmp_path, 'calls-test') as store:
            backend = LLMBackend(cfg, 'test-placeholder')
            policy = NanoPolicy(backend, store)
            count = 0
            async def handler(request):
                nonlocal count
                count += 1
                return httpx.Response(200, json={'id': 'fake', 'model': 'gpt-5.4-nano',
                    'choices': [{'message': {'content': '{"candidates":[1,2,3,4]}'}, 'finish_reason': 'stop'}],
                    'usage': {'prompt_tokens': 100, 'completion_tokens': 12}})
            await backend._client.aclose()
            backend._client = httpx.AsyncClient(base_url='https://test.invalid/v1', transport=httpx.MockTransport(handler))
            view = {'task': 'needle', 'batch_size': 4}
            context = DecisionContext('test', 'e', 0, 1, 'needle', 8)
            one = await policy.decide(view, context)
            # Crash after response persistence, before decision persistence.
            (store.path / 'decisions' / (one['cache_key'] + '.json')).unlink()
            backend.responses.clear()
            two = await policy.decide(view, context)
            assert one == two and count == 1
            await backend.aclose()
    asyncio.run(check())


def test_full_execution_gate(tmp_path, capsys):
    from src.agent_system.no_transition.cli import main
    with pytest.raises(SystemExit):
        main(['--mode', 'full', '--run-dir', str(tmp_path)])
    assert not list(tmp_path.iterdir())


def test_permanent_api_failure_is_redacted_and_cached(tmp_path):
    async def check():
        with RunStore(tmp_path, 'failed-api') as store:
            backend = LLMBackend(LLMConfig(**config()['llm']), 'placeholder')
            count = 0
            async def handler(request):
                nonlocal count
                count += 1
                return httpx.Response(401, text='provider echoed sensitive-test-value')
            await backend._client.aclose()
            backend._client = httpx.AsyncClient(base_url='https://test.invalid/v1', transport=httpx.MockTransport(handler))
            policy = NanoPolicy(backend, store)
            context = DecisionContext('test', 'failure', 0, 1, 'needle', 8)
            first = await policy.decide({'task': 'needle', 'batch_size': 4}, context)
            second = await policy.decide({'task': 'needle', 'batch_size': 4}, context)
            assert first == second and first['api_error'] and first['action'] == {}
            assert count == 1 and first['error']['http_status'] == 401
            for path in store.path.rglob('*.json*'):
                assert 'sensitive-test-value' not in path.read_text()
            await backend.aclose()
    asyncio.run(check())


def test_report_and_archived_reanalysis_need_no_api(tmp_path):
    from src.agent_system.no_transition.report import analyze
    c = config()
    c['deterministic_seeds'] = [7]
    c['bootstrap_samples'] = 20
    for condition in c['conditions']:
        for degree in c['degrees']:
            asyncio.run(run_episode(c, condition, 'deterministic', 7, degree, tmp_path / 'run'))
    output = tmp_path / 'out'
    summary = analyze(tmp_path / 'run', output, c)
    assert not summary['complete'] and summary['episodes'] == 12
    assert 'INCOMPLETE' in (output / 'pilot_report.md').read_text()
    assert len(list((output / 'figures').glob('*.png'))) == 3
    assert len(list((output / 'figures').glob('*.pdf'))) == 3
    regenerated = analyze(output, tmp_path / 'reanalyzed', c)
    assert regenerated['rows'] == summary['rows']
    assert regenerated['diagnostics'] == summary['diagnostics']
