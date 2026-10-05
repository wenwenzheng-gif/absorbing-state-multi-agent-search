# Deterministic model and reproducibility

## 1. Scientific question

How much communication is needed to make incorrect search lineages subcritical in a population of agents that branch hypotheses, test evidence, and exchange only their own experimental results? The observable is the critical graph degree $d_c$ at which the reproduction factor of already-incorrect lineages crosses one.

## 2. Model

Each episode uses one fixed random simple undirected $d$-regular graph on $N$ agents. Agents perform a depth-synchronous confluent search over partial component assignments. The exact round order is

1. receive the previous round's messages and prune;
2. branch every surviving hypothesis;
3. allocate and perform experiments, then prune locally;
4. store the resulting own-experiment messages for delivery next round.

Received evidence is never forwarded. Empty frontiers remain empty. All stochastic choices use explicit, domain-separated seeded streams.

## 3. Parameters

| Symbol | Meaning |
|---|---|
| $N$ | number of agents |
| $M$ | number of globally identified components |
| $K$ | possible values per component |
| $T$ | number of distinct components in the hidden task; complete hypothesis depth |
| $b$ | maximum children sampled per surviving parent per round |
| $m$ | budget-density parameter; total system budget per round is $Nm$ |
| $n_0$ | Mode-B initial frontier width per agent |
| $d$ | degree of the undirected regular communication graph |

There are $T-1$ rounds because Mode-B starts at depth one.

## 4. Hypotheses

A hypothesis is a partial assignment $\lbrace (j, a_j) \rbrace$ on distinct components. Its representation is canonical and unordered: paths that assign the same component-value set in different orders merge into one node. Every agent starts with $n_0$ one-component hypotheses: exactly one true component-value pair and $n_0-1$ distractors.

## 5. Evidence

An experiment tests a globally identified pair $(j, a)$. A positive outcome fixes component $j$ to $a$; a negative outcome removes $a$ from the allowed values of $j$. The outcome is noiseless, and every hypothesis inconsistent with accumulated evidence is immediately removed. A pair has the same meaning for every agent.

## 6. System-wide experiment budget

The round budget is $Nm$. A seeded randomized round-robin allocator repeatedly permutes all agents and gives the next slot to an agent that currently has a valid, untested proposed pair. Candidate sets are recomputed after every experiment because pruning can change availability. The allocator stops only when the global budget is exhausted or no candidate remains. Thus $m$ is not a local quota.

## 7. Communication timing

At round $t$, an agent receives only neighbors' own experiments from round $t-1$. Evidence learned from a neighbor is used for pruning but is not copied into the recipient's outgoing message. Experiments performed at round $t$ become available to neighbors only at round $t+1$.

## 8. Incorrect-lineage reproduction

Let $A^{\mathrm{post}}_{e,t-1}$ be the incorrect population in episode $e$ after local pruning in the preceding round, with the Mode-B initial incorrect population at $t=1$. Let $A^{A\to A}_{e,t,\mathrm{pre}}$ count incorrect children whose direct parent was already incorrect, measured after receive/branch and before local experiments. Newly created incorrect source lineages are excluded from this numerator. The pooled reproduction observable is

```math
\log R_{\mathrm{lineage}}(d)
=\sum_{t=1}^{T-1}
\log\left(
\frac{\sum_e A^{A\to A}_{e,t,\mathrm{pre}}}
     {\sum_e A^{\mathrm{post}}_{e,t-1}}
\right).
```

## 9. Critical degree

The scan finds adjacent integers $d_- < d_+ = d_- + 1$ with $\log R(d_-) > 0 > \log R(d_+)$. The estimator is linear interpolation in $\log R$:

```math
d_c = d_- - \frac{\log R(d_-)}{\log R(d_+) - \log R(d_-)}.
```

Uncertainty is obtained by paired nonparametric bootstrap of the two endpoint trajectories, using identical episode seeds at $d_-$ and $d_+$.

## 10. Analytic theory

The exact finite-$T$ mean-field prediction used in every comparison is computed as

```math
d_c^{\mathrm{analytic}}
= -\frac{(T-1)\ln b}
{\sum_{t=1}^{T-1}\ln\left(1-\frac{mt}{MK}\right)}.
```

In the dilute regime $mT/(MK) \ll 1$, it reduces to

```math
d_c \sim \frac{2MK\ln b}{mT}.
```

The asymptotic expression is documented for intuition; final numerical comparisons use the exact formula.

## 11. Frozen clean regime

| Condition | Requirement |
|---|---|
| SUPPLY-CLEAN | budget utilization $\ge 0.85$, message availability $\ge 0.85$, candidate-capacity ratio $\ge 0.90$, frontier-empty fraction $\le 0.25$, excess frontier Jaccard $\le 0.05$ |
| INDEPENDENCE-CLEAN | duplicate-evidence fraction $\le 0.20$, evidence excess Jaccard $\le 0.30$, $d_c/(N-1) \le 0.40$ |
| THEORY-CLEAN | both conditions above hold |

The machine-readable thresholds are centralized in `configs/clean_criteria.json`; theory error is unavailable when labels are frozen.

## 12. Current result

For the 16 primary THEORY-CLEAN $b=2$ cells,

```math
\boxed{
d_c = 23.614
\left(\frac{M}{16}\right)^{1.079}
\left(\frac{K}{4}\right)^{0.904}
m^{-1.009}
\left(\frac{T}{4}\right)^{-1.489}}
```

with $R^2_{\log} = 0.9990$. The clean data identify $\alpha_M$, $\alpha_K$, $\alpha_m$ and $\alpha_T$, but not $\alpha_b$: all primary THEORY-CLEAN cells have $b=2$. Fits using the redundant $b=3,4,5$ cells are retained only as descriptive diagnostics.

## 13. Reproduce

```bash
python -m pip install -r requirements.txt -r requirements-agents.txt
python reproduce.py --test
python reproduce.py --quick
```

Python 3.12 with the pinned versions in `requirements.txt` is recommended; the NetworkX pin matters because the seeded regular-graph generator is part of the trajectory definition. Conda users can run `conda env create -f environment.yml` instead.

`--test` runs the complete unittest suite, including the allocator tests, golden trajectory hashes, same-seed determinism, a different-seed negative control, the adjacent-integer crossing test, the exact-theory test, and final-number regressions.

Quick mode starts from `data/master_resolved_cells.csv`, recomputes exact theory, refits all reported relations, and redraws the six figures under `reproduced/quick/`. Every regenerated table must match the saved table to $10^{-12}$, and every figure must render; `reproduced/quick/validation.json` records the comparison.

The complete simulation regenerates every raw trajectory under `reproduced/full/raw/`. It is expensive, cache-aware and resumable, and it aborts if a scan does not reproduce the frozen adjacent bracket or a final $d_c$ differs from the deterministic reference:

```bash
python reproduce.py --full --workers 8
```

The frozen inputs are `configs/final_parameter_registry.{csv,json}` (all 53 cells, scan degrees, endpoint sizes and seeds), `configs/seeds.json`, `configs/clean_criteria.json` (the only clean-threshold definition), `data/bootstrap/*.npz`, and `data/raw_data_manifest.csv`.



## Documentation

Run commands from the repository root. The additional agent dependencies are needed
by the complete test suite; deterministic quick/full reproduction only needs
`requirements.txt`. For the LLM protocol, see [LLM experiments](llm_experiments.md).

No full simulation is needed to inspect the included processed results.
