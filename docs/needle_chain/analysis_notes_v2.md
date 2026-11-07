# V2 post-execution analysis notes

All 516 v2 episodes were executed at the preregistered commit
`bd94e66c7b65edd797f6751f0ab30d132187f351`, in hosted run
[42415318874](https://github.com/wenwenzheng-gif/absorb-state-new/actions/runs/42415318874).
The configuration, provider lock, preregistration, environments, policies and
runner still match their pre-execution SHA-256 hashes in `v2_freeze.json`.

After execution, only derived reporting and figure presentation were extended:

- Report the first-round coverage and discovery counts and distinguish these
  from hypothetical cases where the second pass rescues an omission.
- Count changed batches and their overlap with already received negative
  neighbor evidence. The seven changed High batches occurred at d=5 in round 2,
  after all 18 High episodes had complete coverage and success. They retested
  56 neighbor-reported negative IDs. No model motive is inferred.
- Explicitly state that no observed High success required the second pass in
  this pilot. Resource slack, two-pass scheduling and stronger guidance changed
  together, so their separate effects are not identified.
- Add per-panel GPT sample counts and move the comparison-figure legends
  outside the data area.

The preregistered metrics, thresholds and READY decision rule did not change.
V1 was not discarded, overwritten, pooled with v2 or treated as statistically
corrected. All 22 protected v1 files and all 2,884 existing local v1 raw files
remain unchanged. All 3,240 downloaded v2 raw episode files and the processed
episode archive match the hosted artifact exactly. No extra model call,
selective rerun, prompt repair, changed seed or additional resource increase
was made after observing v2.

READY FOR FULL RUN is an operational readiness conclusion. With only six
paired High seeds, absence of a degree effect remains statistically
inconclusive. No full run was authorized or launched.
