import asyncio
import copy
import json

import pytest

from src.agent_system.no_transition.config import ROOT, load_config, expected_episode_count, request_bound
from src.agent_system.no_transition.environments import NeedleEnvironment, public_schedule, reference_batch
from src.agent_system.no_transition.full_analysis import (needle_record, mechanism_cells, deterministic_gate,
    crossover_classification, scientific_conclusion, historical_comparison, compact)
from src.agent_system.no_transition.preservation import verify_protected_pilots, guard_full_paths
from src.agent_system.no_transition.policies import ReferencePolicy
from src.agent_system.no_transition.runner import run_episode


def full():
    return load_config(ROOT/'configs/needle_chain/full_v2.json')


def test_exact_full_design_and_protected_pilots():
    c=full()
    assert expected_episode_count(c,'deterministic')==4800
    assert expected_episode_count(c,'llm')==288
    assert request_bound(c)==28800
    assert c['deterministic_seeds']==list(range(92000,92200))
    assert c['llm_seeds']==list(range(92000,92012))
    pilot=load_config(ROOT/'configs/needle_chain/pilot_v2.json')
    assert c['llm']==pilot['llm']
    assert c['conditions'][2:]==pilot['conditions'][2:]
    assert verify_protected_pilots()['files_verified']==51
    for label in ['pilot','pilot_v2']:
        with pytest.raises(ValueError):
            guard_full_paths(ROOT/f'runs/needle_chain/{label}',ROOT/'results/needle_chain/full_v2')


def test_full_two_passes_and_target_independence():
    c=full()
    high=c['conditions'][1]
    for seed in c['llm_seeds']:
        schedule=public_schedule(high,seed)
        assert len(schedule)==320
        assert set(schedule[:160])==set(schedule[160:])==set(range(160))
        assert schedule[:160]!=schedule[160:]
        env=NeedleEnvironment(160,seed)
        for target in range(160):
            env._target=target
            assert public_schedule(high,seed)==schedule
    low=c['conditions'][0]
    assert 20*low['horizon']*low['batch_size']/low['candidates']==.04


def test_full_reference_gate_and_neighbor_retest_accounting(tmp_path):
    c=full()
    c['deterministic_seeds']=[92000]
    records=[]
    for condition in c['conditions']:
        for degree in c['degrees']:
            e=asyncio.run(run_episode(c,condition,'deterministic',92000,degree,tmp_path))
            if e['task']=='needle':
                records.append(needle_record(e))
    assert deterministic_gate(tmp_path,c)['episodes']==24
    assert all(r['schedule_departure_fraction']==0 for r in records)
    high=[r for r in records if r['condition']=='Needle-High']
    assert all(r['first_round_coverage']==1 and r['duplicate_fraction']==.5 for r in high)
    zero=next(r for r in high if r['degree']==0)
    dense=next(r for r in high if r['degree']==19)
    assert zero['neighbor_informed_duplicate_fraction'] is None
    assert 0<dense['neighbor_informed_duplicate_fraction']<=1
    assert dense['neighbor_reported_retests']==dense['neighbor_known_negative_retests']+dense['neighbor_known_target_retests']
    assert all(r['round1_fraction_of_successes']==1 for r in mechanism_cells(high))
    missing=next((tmp_path/'episodes').glob('*/episode_result.json'))
    missing.unlink()
    with pytest.raises(ValueError,match='incomplete'):
        deterministic_gate(tmp_path,c)


def test_frozen_crossover_rules_and_no_finite_estimate():
    c=full()
    def classify(ps):
        return crossover_classification([{'degree':d,'p_success':p} for d,p in zip(c['degrees'],ps)],c)
    assert classify([0]*6)=='LOW-SATURATED'
    assert classify([1]*6)=='HIGH-SATURATED'
    assert classify([0,.1,.4,.7,.9,1])=='INTERIOR-CROSSOVER'
    assert classify([.4]*6)=='NO-CLEAR-PATTERN'
    patterns=[{'condition':condition['name'],'classification':('LOW-SATURATED' if i in [0,2] else 'HIGH-SATURATED'),
               'credible_paired_crossover':False,'observed_range':0} for i,condition in enumerate(c['conditions']) for kind in ['deterministic','llm']]
    assert scientific_conclusion(patterns,True,True,True)=='CONSISTENT WITH NO-TRANSITION THEORY'
    patterns[0]['observed_range']=.3
    assert scientific_conclusion(patterns,True,True,True)=='INCONCLUSIVE'
    patterns[0]['credible_paired_crossover']=True
    assert scientific_conclusion(patterns,True,True,True)=='EVIDENCE OF DEGREE-DEPENDENT TRANSITION'
    assert scientific_conclusion(patterns,True,False,True)=='INVALID DUE TO IMPLEMENTATION/PROTOCOL FAILURE'


def test_historical_outcome_is_not_fabricated_task_success():
    rows, manifest=historical_comparison()
    assert sum(r['episodes'] for r in rows)==24
    assert all(not r['actual_task_success_available'] and 'p_success' not in r for r in rows)
    assert all(r['agents']==40 for r in rows)
