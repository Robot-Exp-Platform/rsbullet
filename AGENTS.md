# RsBullet development entry

Read [README](README.md), [source builds](docs/source-build.md), and
[native simulation semantics](docs/native-simulation.md) before choosing an
application architecture. This repository supplies native Rust/Bullet physics;
it does not require a Python/PyBullet subprocess.

- Start automated work with DIRECT and `default-features = false`. Do not launch
  the older interactive examples or run ignored GUI tests unattended.
- Initialize the pinned `rsbullet-sys/bullet3` submodule. Preserve model meshes,
  licenses and revisions. GitHub ZIPs do not contain the submodule closure.
- Respect the documented sibling source revisions and Cargo lockfile. The current
  integration uses explicit local patches until the new crates are published.
- Set `BULLET_SKIP_ASSET_EXPORT=1` and `ROPLAT_SKIP_ASSET_EXPORT=1` in builds.
  Keep build/cache directories outside source and preserve experiment evidence.
- Use finite `SimRhythm` with a real `#[roplat::system]` graph for roplat apps.
  Commands queued by that graph apply at the next integration. Actual state and
  contacts come from the native simulator, not the requested command.
- Select the intended joints by name when a model includes a hand or other
  independent part. State explicitly which frames, limits and collision flags
  the application uses. An unsupported capability must remain explicit.
- After changing code, run the affected finite tests and Clippy. The relevant
  suites are `rsbullet-core/tests/native_contract.rs`,
  `rsbullet/tests/runtime_headless.rs`, and `rsbullet/tests/sim_rhythm.rs`.
  Preserve failures and report untested platform/feature combinations.
- Shared traits belong in `roplat_behavior` only when they express reusable
  capabilities; simulator-specific queries and task controllers remain in their
  respective backend and application layers.
- Library, contract and driver code follows the style rules in roplat-skills: [library-code-style.md](https://github.com/Robot-Exp-Platform/roplat-skills/blob/next/skills/roplat-development/references/library-code-style.md) (branch `next`; every rule carries a strength: must / should / may). New code follows them at once; existing code when it is touched.

The finite `panda_headless` example is an API starting point. It does not prove
grasp success, physical safety, all-device support, or full PyBullet parity.
