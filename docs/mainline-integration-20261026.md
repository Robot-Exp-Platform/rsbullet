# Mainline integration, 2026-10-26

This integration retains the ancestry of PRs #6, #7 and #8. RsBullet/core are
0.7.0; the headless feature split and native fixes use rsbullet_sys 0.3.3.
Both package versions and dependencies remain unreleased source preparation.

The behavior dependency is the sibling four-crate 0.9 workspace with the model
selection and SimulationRhythm changes from behavior #16 retained through #17.
SimRhythm stays an alias. The merge preserves post-integration observations,
link-origin velocity, absent joint measurements, control submission order,
effort limits, seeded IK validation and world/handle invalidation. New examples
and tests use the renamed contracts and mask-based sample accessors.

The root patch table and regenerated Cargo.lock select the sibling sources.
See source-build.md for the required layout and external-consumer patch rules.
The existing add_robot unit test now uses the #7 repository fixture, so it does
not depend on exported robot.urdf. Three atomic updates and a pixel iteration
were updated to their equivalent APIs for nightly-2026-07-07 lint compatibility.

Windows x86_64/MSVC verification with that toolchain, both asset-export flags
set and local integration sources:

- 7 core native tests pass (native_contract and multidof_motor).
- 12 high-level tests pass with roplat enabled; the interactive GUI test remains
  ignored. The shared rhythm executes an actual System graph.
- Headless all-target Clippy for rsbullet and rsbullet-core passes with -D warnings.
- The finite Panda example is rerun for the original A/B/repeat configurations;
  detailed receipts and native traces belong to the integration audit, not Git.

Native linking emits the existing MSVC LNK4098 CRT warning; execution succeeds.
GUI execution and external transport services are outside this validation.
Earlier quantitative trials in pick-place.md retain their original source pins
and do not by themselves validate this new combination.
