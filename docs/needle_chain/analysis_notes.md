# Post-pilot analysis notes

All 24 live episodes ran at `82d6c2bb7eb79b77b069889b3d563c9122640069`.
The following analysis/report changes were made after observing the saved
pilot, without new API calls or modifications to episodes, prompts, resource
configs, environments or success criteria.

- Added per-batch comparison of public Needle assignments against actual
  selections. This diagnoses all three Needle-High misses and establishes
  that all 11 changed high-resource batches preceded neighbor evidence.
- Withheld a logistic cluster-bootstrap confidence interval when fewer than
  95% of resamples yield identifiable fits, the conditional distribution is
  degenerate, or there are fewer than six task clusters. The two-seed
  Needle-High sample otherwise gives a misleading conditional [0, 0] interval
  after dropping separated resamples. Conditional quantiles and fit counts
  remain available; endpoint intervals, raw outcomes and likelihoods are
  unchanged. These conservative reporting guards are analysis safeguards,
  not preregistered inferential thresholds or evidence of equivalence.
- Added a round-batch runtime scenario and a degree-context token-cost
  extrapolation. Both are explicitly assumption-dependent estimates from
  measured throughput, not observed full-run results.
- Moved success-figure legends outside the plotting areas so they cannot
  obscure the observed Needle-High probability of 0.5.

All 480 deterministic episode JSON values match their pre-hosted hashes.
All 24 live results, including the three unexpected high-resource failures,
are retained. No additional samples, prompt repairs or favorable reruns were
used. The completed full test suite passes; see the pilot's `validation.json`.
