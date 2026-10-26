# Building and deploying the source branch

RsBullet is a native Rust/Bullet integration. Python and PyBullet are not build
or runtime dependencies. The 0.7.0 source APIs are under development; the older
registry examples do not select this code.

This branch prepares `rsbullet` / `rsbullet-core` 0.7.0 and `rsbullet_sys` 0.3.3.
The new sys features require this source revision; published sys 0.3.2 does not
contain them. This is Git release preparation, not a crates.io release.

## Source closure

Keep the RsBullet revision, `roplat_behavior` revision, roplat revision (when
used), Bullet submodule revision, model files, Cargo.lock and Rust toolchain
together in the experiment record. A source ZIP alone omits the Bullet source.
Set `RSBULLET_REVISION` to the reviewed commit or release from that record;
the repository's default branch may still contain an earlier implementation.

```sh
git clone https://github.com/Robot-Exp-Platform/rsbullet.git
cd rsbullet
git checkout "$RSBULLET_REVISION"
git submodule update --init --recursive rsbullet-sys/bullet3
git rev-parse HEAD
git submodule status rsbullet-sys/bullet3
```

The unmerged integration uses sibling source checkouts because
`roplat_behavior` / `roplat_trajectory` 0.9 and `roplat_msgs` are not released.
Arrange reviewed revisions as follows:

```text
workspace/
  roplat/             # core, including roplat_native and roplat_msgs
  roplat_behavior/    # four-crate workspace, including SimulationRhythm
  rsbullet/          # this repository
```

The root `Cargo.toml` patches the registry package names to those siblings;
`Cargo.lock` must be generated and checked for this exact combination. Do not
replace these sources with old published `robot_behavior` or with an arbitrary
floating main. Record `git rev-parse HEAD` for all three checkouts and the
Bullet gitlink alongside test results. The integration commits are local until
explicitly pushed; a version number does not imply registry availability.

Cargo ignores a dependency's patch table when building a separate application.
Such an application's root must provide the same source mapping, for example:

```toml
[patch.crates-io]
roplat = { path = "../roplat/roplat" }
roplat_msgs = { path = "../roplat/roplat_msgs" }
roplat_behavior = { path = "../roplat_behavior/roplat_behavior" }
roplat_trajectory = { path = "../roplat_behavior/roplat_trajectory" }
```

Adjust paths for the application's location. Use authenticated host clones or
credential-free source snapshots for containers; do not copy host credentials
into an image. `--locked` verifies the saved resolution, while `--offline` also
requires every registry index entry and archive to be cached.

## Choose the build features

| Build | Requirements | Intended use |
| --- | --- | --- |
| `--no-default-features` | C++ compiler, CMake; Linux Fontconfig dev files for `roplat_trajectory` | CPU DIRECT simulation, URDF, IK, contacts and TinyRenderer |
| Default (`gui`) | Above plus platform OpenGL/window development libraries | Native Bullet GUI or in-process graphics server |
| `--no-default-features --features roplat` | Headless requirements plus roplat source | Finite or paced task graph integration |
| `enet` / `clsocket` | Matching native UDP / TCP code compiled by the build script | Explicit remote connection modes |

Enabling a transport compiles its client implementation; it does not start or
validate a remote service. The native BulletRobotics archive includes the ENet
and clsocket implementation sources when those options are enabled. Live
network interoperability must be tested separately from a DIRECT task.
The GUI build also needs Bullet's internal TCP helper; this does not enable the
public Rust TCP connection mode, which still requires `clsocket` explicitly.

Cargo unifies dependency features. If another crate enables `rsbullet-core/gui`
or `rsbullet_sys/gui`, the native build will include GUI support even when your
direct `rsbullet` dependency says `default-features = false`. Inspect
`cargo tree -e features` when preparing a server image.

`Mode::Direct` selects a runtime connection. The `gui` Cargo feature selects
which native code is built. Asking for a GUI mode in a headless build returns an
error; it never silently changes to DIRECT. An unavailable GUI session can still
fail when GUI support was compiled.

Linux headless prerequisites:

```sh
sudo apt-get update
sudo apt-get install build-essential cmake ninja-build pkg-config libfontconfig1-dev
rustup toolchain install nightly-2026-07-07 --profile minimal
BULLET_SKIP_ASSET_EXPORT=1 ROPLAT_SKIP_ASSET_EXPORT=1 \
  cargo +nightly-2026-07-07 test -p rsbullet-core --no-default-features --lib --tests
BULLET_SKIP_ASSET_EXPORT=1 ROPLAT_SKIP_ASSET_EXPORT=1 \
  cargo +nightly-2026-07-07 test -p rsbullet --no-default-features --features roplat --lib --tests
```

The optional [Dockerfile](../docker/Dockerfile.headless) prepares the same
compiler/toolchain class. It intentionally installs no OpenGL/X11 development
packages. Pin the Ubuntu image digest and archive package/tool versions when a
comparison requires an immutable environment; the recipe alone is not a
bit-for-bit build guarantee.

```sh
docker build -f docker/Dockerfile.headless -t rsbullet-headless:dev docker
```

After the headless tests, run a finite Panda application from the repository
root. It writes actual end-effector feedback and motor effort to a CSV:

```sh
BULLET_SKIP_ASSET_EXPORT=1 ROPLAT_SKIP_ASSET_EXPORT=1 \
  cargo +nightly-2026-07-07 run -p rsbullet --no-default-features \
  --example panda_headless -- \
  rsbullet-sys/bullet3/examples/pybullet/gym/pybullet_data panda-trace.csv
```

On Docker Desktop, prefer a Linux filesystem or named volume for Cargo's
registry unpack directory and build target. Writing thousands of compiler
intermediates through a Windows bind mount can dominate build time. Keep logs,
source fingerprints and final outputs in the host artifact directory. Report
whether registry archives or compiler outputs were reused; a fresh container
with a seeded Cargo cache is not a cold installation measurement.

## Host and simulator support are separate

Headless Bullet needs a supported CPU/compiler and the model's physical
description. A display or GPU is unnecessary for DIRECT and TinyRenderer.
Hardware acceleration, vendor device support, and model/control semantics are
separate capabilities. A Panda URDF running in Bullet does not validate a Franka
hardware driver or prove that the same model works in a different simulator.

Native Windows needs MSVC C++ Build Tools and the Windows SDK; macOS needs Xcode
Command Line Tools. Both need CMake. The source build script uses the Cargo
target OS for system-library selection, but a cross-compilation toolchain is
still the caller's responsibility. Linux amd64, Windows native and macOS/ARM
must be reported as distinct validation targets, never inferred from one build.

## Resources and failures

Use `BULLET_SKIP_ASSET_EXPORT=1` and `ROPLAT_SKIP_ASSET_EXPORT=1` in automated
builds. Supply model roots explicitly; keep URDF-relative mesh paths intact.
The bundled Bullet data can be used directly from
`rsbullet-sys/bullet3/examples/pybullet/gym/pybullet_data`.

The build reports an actionable error when Bullet's CMakeLists.txt is absent.
If native compilation fails, keep the CMake/compiler output rather than treating
it as a simulator API failure. Test a finite DIRECT executable after compiling:
`cargo check` alone does not test native linkage, stepping, or robot motion.

The sys build places static libraries under Cargo's native build directory and
links them directly. It does not install Bullet's complete C++ SDK/header tree;
applications needing that SDK should build/install Bullet explicitly. This
avoids scanning and copying unrelated example assets during a Rust build.
