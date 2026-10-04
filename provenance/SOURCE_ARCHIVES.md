# Source archives and consolidation decisions

The package was assembled from the following immutable project artifacts:

| Historical artifact | SHA-256 | Use |
|---|---|---|
| `dc_scaling_clean_regime_v2_final.tar.gz` | `5f2245e794c8cf11eeb72d5ff1c1f91561afc6c0e91ddeb27502e72647b1942b` | v2 trajectories, registry, bootstrap sources |
| `dc_scaling_neighbor_independence_v3_final.tar.gz` | `149e106c6807ae463ec65cd95660d5d4de96ca2416cc78cac57ebe7627e58d6d` | width/independence validation and v2 replay checks |
| `dc_scaling_logb_validation_final.tar.gz` | `54ec52edeb82fe9240fe91457ef368a4e67e1c7e3bac37827eabdd325129bcfb` | matched-(\ln b) data, paired draws, canonical validated simulator |
| `dc_scaling_complete_validation_final.tar.gz` | `f1424eb78c5f777af9d436973432d6177e34b3f1152a9ebf5a81380073fa60cd` | final 53-cell synthesis, tables, figures |
| `code_snapshot_20260917T223054Z.tar.gz` | `85d10f5e6816c4dde155532a9e78fe5eec0d827e76bd94f2a1462c3aa7cc1b40` | latest b-clean expansion code and regression tests |

## Canonical simulator selection

The selected `src/scaling_model.py` has SHA-256
`3bc299775da0e9877e0062e7cf73c2cdc8f254c74c156c83c62e4b8eccb770e8`.
It is byte-identical in the latest b-clean expansion and the completed
matched-(\ln b) project. Relative to the v2/v3 file
(`a15c1440ae4520856e33270068918f90ab5c768d45fb2470c64cc934f0bc520e`),
the existing system-wide round-robin loop was extracted into a named helper
and read-only budget diagnostics were added. No scientific transition or RNG
draw changed. Eight allocator/regression tests pass, including three
pre-change golden trajectory hashes after projecting away the new diagnostic
fields.

The high-(MK/m) b-clean expansion snapshot contained a frozen registry and
code but no completed, validated result tables. No newer final b-clean result
artifact was available at packaging time. It therefore supplies the canonical
code but does not alter the final numerical fit; (alpha_b) remains explicitly
unidentified by THEORY-CLEAN data.

## Removed historical duplication

The package omits the five legacy directory trees, repeated
simulator copies, caches, raw trajectories, checkpoint writers, recovery
scripts, artifact-preview scripts, and historical report variants. They were
removed because they encode project history rather than a distinct production
dynamics. Their scientific content is represented by one simulator, one
registry, one analysis path, compact bootstrap draws, final processed data,
and the hashes above.

`FILE_HASHES.sha256` is preserved unchanged from the original supplementary
package. It hashes every other file in that original package and omits itself,
whose inclusion would be self-referential. Entries for documentation changed
in the public repository describe the original bytes.

`PUBLIC_FILE_HASHES.sha256` covers the public-review repository, including the
original manifest, and omits only itself. Use it to verify the current files;
the original manifest remains available as a historical provenance record.
