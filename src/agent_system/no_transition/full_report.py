"""Derived full report; all scientific decisions are frozen in preregistration."""
from collections import defaultdict
import json

from .analysis import mean


def fmt(x):
    if x is None:
        return 'NA'
    if isinstance(x, float):
        return f'{x:.5g}'
    if isinstance(x, list):
        return '[' + ', '.join(fmt(v) for v in x) + ']'
    return str(x)


def table(headers, rows):
    return '\n'.join(['| '+' | '.join(headers)+' |', '| '+' | '.join(['---']*len(headers))+' |']+
                     ['| '+' | '.join(fmt(v) for v in row)+' |' for row in rows])


def render_full(config, rows, mechanisms, episodes, diagnostics, patterns, historical, runtime, audit, conclusion, sha):
    llm=[e for e in episodes if e['agent_type']=='llm']
    high=[e for e in llm if e['condition']=='Needle-High']
    successes=sum(e['success'] for e in high)
    first=sum(e['round_of_success']==1 for e in high)
    second=sum(e['round_of_success']==2 for e in high)
    parts=[f'''# Needle / Chain full v2 experiment

**Scientific classification: {conclusion}.** This is the full experiment, not a pilot.

`FULL_V2_PREREGISTRATION_SHA = {sha}`

## Objective and frozen implementation

Test whether verified task success shows an interior low-to-high crossover across communication degree in Needle and Chain Search. The [theoretical reference](https://arxiv.org/abs/2609.38327) predicts no positive finite communication threshold for Needle and a sequential resource constraint for Chain. The theoretical labels are Needle d_c=0, Chain-Short d_c=∞, and Chain-Sufficient d_c=0; these labels were not estimated from flat curves.

N=20; degrees 0,4,8,12,16,19. Deterministic seeds 92000–92199 (200/cell); GPT seeds 92000–92011 (12/cell), paired across every degree. Target, schedule and Chain RNG domains are separate; graph seed is task seed +400000. All task resources and prompts are independent of degree. Degree changes only graph-mediated, next-round neighbor observations.

| Condition | Space / length | Horizon | Batch | Nominal verification slots |
| --- | --- | --- | --- | --- |
| Needle-Low | S=4000 | 2 | 4 | 160 (4% reference coverage) |
| Needle-High | S=160 | 2 | 8 | 320 (two complete public passes) |
| Chain-Short | C=8 | 6 | 1 | 120 |
| Chain-Sufficient | C=8 | 10 | 1 | 200 maximum; 160 if completed at round 8 |

Model `gpt-5.4-nano-2026-03-17`; OpenAI chat completions, temperature 1, reasoning effort none, max output 256, concurrency 6, timeout 60 s, up to two transport retries, zero repair attempts, no per-agent API seed. The unchanged v2 High prompt recommends its schedule, while the schema permits other legal candidates. Needle success means the target was actually tested; Chain success means level 8 was completed by the horizon. Chain advancement is at most one level per global round. Needle runs both rounds even after first discovery.

See `config.json`, `provider_settings.json`, `full_v2_freeze.json`, `data_audit.json`, `validation.json`, `hosted_provenance.json` and the frozen preregistration for exact settings and checks. Completed calls and episodes are checkpointed; no behavioral failures are selectively rerun. All scheduled outcomes, including API-error episodes, remain in denominators.

## Primary success probabilities

The table directly reports P_success(d); intervals are marginal 95% Wilson intervals. D=deterministic, G=GPT. Mean rounds means rounds executed, including both Needle passes; API error is the fraction of episodes with at least one terminal model-decision error, distinct from retried transport attempts.
''']
    parts.append(table(['Condition','Agent','d','n','Successes','P_success','95% CI','Mean rounds','Invalid rate','API-error fraction'],
        [[r['condition'], 'D' if r['agent_type']=='deterministic' else 'G',r['degree'],r['episodes'],r['successes'],r['p_success'],[r['ci_low'],r['ci_high']],r['mean_rounds_used'],r['invalid_action_rate'],r['api_error_episode_fraction']] for r in rows]))
    parts.append('''
## Paired endpoints and degree-effect models

ΔP=P_success(19)−P_success(0). The bootstrap resamples whole seed pairs (1,000 replicates). The conservative interval subtracts Bonferroni 97.5% marginal Wilson intervals; it remains informative about uncertainty when a paired bootstrap collapses to [0,0]. McNemar exact discordance tests and estimable cluster-robust logistic slope tests form one BH family. A null result is not proof of equality. Logistic likelihood/AIC are descriptive Bernoulli working-model comparisons, with uncertainty clustered by task seed.
''')
    parts.append(table(['Condition','Agent','Pairs','ΔP','Bootstrap 95%','Conservative 95%','Discordant','p','BH q'],
        [[r['condition'],r['agent_type'],r['endpoint'].get('paired_seeds'),r['endpoint'].get('delta_p'),r['endpoint'].get('ci95_paired_bootstrap'),r['endpoint'].get('ci95_conservative_wilson'),r['endpoint'].get('discordant_pairs'),r['endpoint']['p_value'],r['endpoint'].get('p_bh_fdr')] for r in diagnostics]))
    parts.append(table(['Condition','Agent','Slope / degree','Cluster SE','Cluster bootstrap 95%','LL constant / degree','AIC constant / degree','p / q','Status'],
        [[r['condition'],r['agent_type'],r['regression'].get('coefficient'),r['regression'].get('cluster_robust_se'),r['regression'].get('ci95_cluster_bootstrap'),[r['regression'].get('log_likelihood_constant'),r['regression'].get('log_likelihood_degree')],[r['regression'].get('aic_constant'),r['regression'].get('aic_degree')],[r['regression']['p_value'],r['regression'].get('p_bh_fdr')],r['regression']['status']] for r in diagnostics]))
    parts.append('For constant outcomes, degree slope is not identifiable because outcome is constant. No finite slope, inferential p-value or finite critical degree is manufactured. Conditional bootstrap estimates and reasons for unreliable intervals are retained in `diagnostics.json`. Twelve GPT clusters provide limited inferential precision.')
    parts.append('## Explicit crossover diagnostic\n\nPreregistered descriptive low≤0.2 and high≥0.8 thresholds classify the observed curves. Credible crossover additionally requires at least 75% paired seeds changing failure→success between an ordered low/high degree pair and exploratory BH q≤0.05 across all ordered-pair tests. This is distinct from the primary endpoint/slope BH family. No finite d_c is fitted.')
    parts.append(table(['Condition','Agent','Classification','Observed range','Credible paired crossover','Brackets'],
        [[r['condition'],r['agent_type'],r['classification'],r['observed_range'],r['credible_paired_crossover'],r['credible_brackets']] for r in patterns]))
    parts.append('''## Needle mechanism results

Unique coverage counts actual valid tests. Duplicate fraction counts repeats over all valid tests, including the deliberate second High pass. Schedule departures use multiset differences, so merely reordering a batch is not a departure. Target-batch following is the fraction of target-containing scheduled batches whose complete multiset was executed; actual target testing is reported separately. Neighbor retests count valid tests of a candidate already reported in the agent's received evidence for that round. The neighbor fraction uses valid tests made with nonempty neighbor candidate evidence as denominator; NA means no evidence. These are observable associations, not inferred model motives or causal communication effects.
''')
    parts.append(table(['Condition','Agent','d','Unique','Coverage','Duplicates','Dup. fraction','Departure fraction','Target batch followed','Target tested'],
        [[r['condition'],r['agent_type'],r['degree'],r['mean_unique_candidates_tested'],r['mean_coverage_fraction'],r['mean_duplicate_tests'],r['mean_duplicate_fraction'],r['mean_schedule_departure_fraction'],r['target_batch_followed_fraction'],r['target_tested_episode_fraction']] for r in mechanisms]))
    parts.append(table(['Condition','Agent','d','Neighbor retests','Known negative','Known target','Neighbor fraction','Round 1 / episodes','Round 2 / episodes'],
        [[r['condition'],r['agent_type'],r['degree'],r['neighbor_reported_retests'],r['neighbor_known_negative_retests'],r['neighbor_known_target_retests'],r['neighbor_informed_duplicate_fraction'],r['round1_success_fraction_of_all_episodes'],r['round2_success_fraction_of_all_episodes']] for r in mechanisms]))
    parts.append(f'''### Needle-High positive control

GPT successes: {successes}/{len(high)}; round 1 discoveries: {first}; round 2 discoveries: {second}. Among successes, round 1 fraction={fmt(first/successes if successes else None)}, round 2 fraction={fmt(second/successes if successes else None)}. The second pass rescued {second} first-round omissions ({fmt(second/len(high) if high else None)} of episodes). Per-degree fractions are in `needle_mechanisms.csv`; discovery rounds for every episode are in `needle_episodes.csv`.

This is a saturated positive-control regime showing whether success remains high throughout the communication-degree scan when resources are sufficient. Smooth coverage or duplication changes are compatible with no phase transition.''')
    if high and first==len(high):
        parts.append("The high-resource condition is saturated before communication feedback becomes relevant and therefore serves primarily as a positive control rather than an identification of communication's causal effect.")
    parts.append('## Chain mechanism results\n\nDepth trajectories are exported in `depth_by_round.csv` and Figure 4. Completed depth remains absorbing for display after early termination. Each observed round is checked for an increase of at most one; future-key guesses cannot bypass the level lock.')
    groups=defaultdict(list)
    for e in episodes:
        if e['task']=='chain':
            groups[e['condition'],e['agent_type'],e['degree']].append(e)
    parts.append(table(['Condition','Agent','d','Mean max depth','Max depth','Illegal future attempts','Mean completion round (successes)'],
        [[*key,mean(e['maximum_depth'] for e in es),max(e['maximum_depth'] for e in es),sum(e['illegal_future_attempts'] for e in es),mean(e['round_of_success'] for e in es if e['success']) if any(e['success'] for e in es) else None] for key,es in sorted(groups.items())]))
    parts.append('''## Historical Hierarchical comparison

No historical model calls were made. The archived source has 24 deterministic episodes, N=40, T=8, degrees 0,1,2,3,4,5,6,8 and seeds 95000–95002, from `wenwenzheng-gif/absorb-state-new`, commit `236a8e043c433239b03ef809f607b365fc933c81`. It initializes only wrong depth-one prefixes. It records incorrect-lineage extinction and growth, not verified target recovery. Execution `status: ok` is not task success. Thus a reproducible historical P_success curve is unavailable from these records; the existing success-model utilities cannot recover an outcome never recorded.

Figure 5A honestly displays P(wrong lineage extinct by T=8), not P_success. This is a mechanism comparison, not evidence that historical task success has the same crossover. The archived lineage-growth analysis reports a crossing between d=3 and d=4 (interpolated 3.599); that is a different estimand. Panels B/C show actual new GPT task success, on separate N=20 degree axes. The older N=40 factorizable threshold tables are also not substituted for Hierarchical success. Source hashes and limitations are in `historical_provenance.json` and `data/needle_chain/historical_hierarchical/source_manifest.json`.
''')
    parts.append(table(['Historical d','n','Extinct','P(extinction)','Wilson 95%'],[[r['degree'],r['episodes'],r['extinct'],r['p_extinction'],[r['ci_low'],r['ci_high']]] for r in historical]))
    parts.append('## Figures\n\nPNG previews and vector PDF versions share the same numerical sources.')
    for name,caption in [('figure_1_needle','Needle success'),('figure_2_chain','Chain success'),('figure_3_needle_mechanisms','Needle coverage and duplication'),('figure_4_chain_depth','Chain depth by global round'),('figure_5_historical_comparison','Historical extinction versus new verified task outcomes')]:
        parts.append(f'![{caption}](figures/{name}.png)\n\n[{caption}, vector PDF](figures/{name}.pdf)')
    parts.append('## Runtime, calls and tokens\n\nMeasured stage wall times exclude dependency installation, tests between stages, artifact uploads and offline analysis. Transport latency is not summed as wall time because calls are concurrent. Estimated cost uses frozen rates $0.20/M uncached input, $0.02/M cached input, $1.25/M output; it is not a billing statement.')
    parts.append(table(['Quantity','Measured value'],[[k,v] for k,v in runtime.items()]))
    parts.append(f'''## Validation, deviations and limitations

Expected episodes: {audit['expected_episodes']}; observed: {audit['episodes']}; missing: {len(audit['missing_episodes'])}. Protected pilot artifacts: {audit['protected_pilots']['status']}; frozen design: {audit['frozen_design']['status']}. Response models: {', '.join(audit['response_models']) or 'none yet'}. Detailed request-key, retry, source-SHA and raw-checkpoint audits are in the machine-readable validation/provenance files.

No planned task parameters, seeds, sample sizes, prompts, success criteria or model settings were changed after execution began. No exclusions or favorable behavioral reruns are permitted. The historical panel uses extinction because verified historical success is unavailable; this limitation was documented before full live execution. Any operational recovery or reporting-only corrections are recorded in `validation.json`.

The deterministic baseline is an implementation and resource check. GPT n=12 per degree cannot prove equality, excludes neither small degree effects nor unobserved transitions between sampled degrees, and is conditional on this policy, graph construction and resource allocation. At 12/12 successes the 95% Wilson lower bound is only about 0.758; at 0/12 the upper bound is about 0.242. No observed crossover is weaker than proof that none exists. Constant outcomes also limit logistic inference. High-resource saturation limits causal interpretation, while the historical panel uses a different outcome and population size.

## Conclusion

**{conclusion}.** The classification applies the frozen regime-separation and paired-crossover rules, rather than selecting outcomes to match theory.''')
    if conclusion=='CONSISTENT WITH NO-TRANSITION THEORY':
        parts.append('Across the preregistered communication-degree scan, the data are consistent with the theoretical no-finite-transition prediction and show no observed interior success crossover. This finite experiment does not prove that no transition exists under other degrees, resources or policies.')
    else:
        parts.append('The evidence and uncertainty above should be interpreted as observed; no additional sampling was initiated to change this classification.')
    return '\n\n'.join(parts)+'\n'
