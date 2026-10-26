# Building and deploying the source branch

RsBullet is a native Rust/Bullet integration. Python and PyBullet are not build
or runtime dependencies. The 0.6.0 source APIs are under development; the older
registry examples do not select this code.

This branch prepares `rsbullet` / `rsbullet-core` 0.6.0 and `rsbullet_sys` 0.3.3.
The new sys features require this source revision; published sys 0.3.2 does not
contain them. This is Git release preparation, not a crates.io release.

## Source closure

Keep the RsBullet revision, `robot_behavior` revision, roplat revision (when
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

The dependency manifests pin Git revisions. Retrieve private first-party source
on the authenticated host; never copy private keys or access tokens into an
agent's container. For a locally developed dependency, use explicit Cargo source
patches and record that the build uses the working source, not the pinned remote
revision. For example, with source mounted at `/src`, `/deps/robot_behavior`,
and `/deps/roplat`, save this as `/config/local-sources.toml`:

```toml
[patch."ssh://git@github.com/Robot-Exp-Platform/robot_behavior.git"]
robot_behavior = { path = "/deps/robot_behavior" }

[patch."ssh://git@github.com/Robot-Exp-Platform/roplat.git"]
roplat = { path = "/deps/roplat/roplat" }
```

Then add `--config /config/local-sources.toml` to Cargo commands. This does not
rewrite the committed dependency declarations. In an integrated development
checkout, do not claim `--locked` reproducibility until the lock has been
resolved for those exact sources and saved with the experiment.

Cargo can still inspect the original pinned Git revision while resolving a
patched dependency. For an isolated agent, prepare source-only bare mirrors on
the authenticated host, containing those exact commits:

```sh
git clone --bare --no-hardlinks /host/robot_behavior /bundle/git-dependencies/robot_behavior.git
git clone --bare --no-hardlinks /host/roplat /bundle/git-dependencies/roplat.git
```

Mount that directory read-only at `/git-dependencies`, then configure the
container's own Git transport (from a directory outside a mounted submodule):

```sh
cd /tmp
git config --global url.file:///git-dependencies/robot_behavior.git.insteadOf \
  ssh://git@github.com/Robot-Exp-Platform/robot_behavior.git
git config --global url.file:///git-dependencies/roplat.git.insteadOf \
  ssh://git@github.com/Robot-Exp-Platform/roplat.git
export CARGO_NET_GIT_FETCH_WITH_CLI=true
```

This retains Cargo's declared source identity while obtaining Git objects from
the supplied mirror. The optional path patches above select current working
code; without those patches Cargo builds the pinned revisions. Neither mode
needs host credentials inside the container. A bare mirror contains repository
history, so share it only with agents authorized for that repository.

## Choose the build features

| Build | Requirements | Intended use |
| --- | --- | --- |
| `--no-default-features` | C++ compiler, CMake; Linux Fontconfig dev files for `robot_behavior` | CPU DIRECT simulation, URDF, IK, contacts and TinyRenderer |
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
rustup toolchain install nightly-2025-11-20 --profile minimal
BULLET_SKIP_ASSET_EXPORT=1 ROPLAT_SKIP_ASSET_EXPORT=1 \
  cargo +nightly-2025-11-20 test -p rsbullet-core --no-default-features --lib --tests
BULLET_SKIP_ASSET_EXPORT=1 ROPLAT_SKIP_ASSET_EXPORT=1 \
  cargo +nightly-2025-11-20 test -p rsbullet --no-default-features --features roplat --lib --tests
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
  cargo +nightly-2025-11-20 run -p rsbullet --no-default-features \
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
