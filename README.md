# RsBullet

RsBullet brings Bullet physics to Rust: load robot models, create bodies, step a simulation, and inspect or control their state. Use it for simulation and algorithm development, or when you want a simulated robot to expose the same `robot_behavior` interfaces used by the driver ecosystem.

It builds the native Bullet engine; a Python interpreter or an installed `pybullet` package is not required. This repository continues the [RuBullet](https://github.com/neachdainn/rubullet) project. The original documentation is preserved in [README_ORIGIN.md](README_ORIGIN.md).

## Choose the right layer

| Crate | Version | Responsibility |
| --- | --- | --- |
| `rsbullet` | `0.4.0` | `RsBullet`, typed robot builders, queued control, and an optional roplat simulation rhythm. Re-exports the core API. |
| `rsbullet-core` | `0.4.0` | `PhysicsClient`: explicit Bullet commands, body/joint IDs, geometry, model loading, and state queries. |
| `rsbullet_sys` | `0.3.2` | Builds the bundled Bullet C++ source and exposes its raw C FFI. |

Most applications should depend on `rsbullet`. The layers keep native build details below the command API, while the higher layer manages robot handles and control callbacks. You can still access `sim.client` when you need a Bullet operation that has no higher-level wrapper.

`RsBullet` owns the client and the simulation clock. A robot's `enqueue` or `control_with` call submits work; it does not run the physics engine. Each `sim.step()` drains queued work, runs active callbacks, then advances Bullet once. Callback completion removes that callback. This makes ownership and stepping explicit, but the application must keep stepping while queued work is active. A simulation step is not a guarantee of a real-time deadline or of matching a physical robot's behavior.

## Prerequisites

Use Rust nightly: `robot_behavior`, a normal dependency, uses unstable Rust features. The native build also needs CMake and a C++ compiler:

- **Windows:** the MSVC C++ build tools, Windows SDK, and CMake on `PATH`.
- **Linux:** C/C++ tools, CMake, OpenGL/GLU, and X11 development libraries. On Ubuntu, a starting set is `build-essential cmake libgl1-mesa-dev libglu1-mesa-dev libx11-dev libxi-dev`.
- **macOS:** Xcode Command Line Tools and CMake; the build links the Cocoa and OpenGL frameworks.

The current native build includes Bullet graphics support even when your program uses `Mode::Direct`. DIRECT avoids opening a simulation window; it does not remove these build dependencies. GUI modes additionally need an available graphical session.

## First simulation: a falling sphere, with no model files

Create a binary project and replace its files as below:

```sh
cargo new rsbullet-demo
cd rsbullet-demo
```

`Cargo.toml`:

```toml
[package]
name = "rsbullet-demo"
version = "0.1.0"
edition = "2024"

[dependencies]
rsbullet = "0.4.0"
robot_behavior = "0.6.1"
nalgebra = "0.34"
anyhow = "1"
```

`src/main.rs`:

```rust
use std::time::Duration;

use nalgebra::Isometry3;
use robot_behavior::PhysicsEngine;
use rsbullet::{
    CollisionGeometry, CollisionId, CollisionShapeOptions, Mode,
    MultiBodyBase, MultiBodyCreateOptions, RsBullet,
};

fn main() -> anyhow::Result<()> {
    let mut sim = RsBullet::new(Mode::Direct)?;
    sim.set_step_time(Duration::from_secs_f64(1.0 / 240.0))?
        .set_gravity([0.0, 0.0, -9.81])?;

    let sphere = sim.client.create_collision_shape(
        &CollisionGeometry::Sphere { radius: 0.1 },
        Some(CollisionShapeOptions::default()),
    )?;
    let body = sim.client.create_multi_body(&MultiBodyCreateOptions {
        base: MultiBodyBase {
            mass: 1.0,
            pose: Isometry3::translation(0.0, 0.0, 2.0),
            collision_shape: CollisionId(sphere),
            ..Default::default()
        },
        ..Default::default()
    })?;

    // Advance 0.5 seconds of simulated time. No wall-clock sleep is needed.
    for _ in 0..120 {
        sim.step()?;
    }
    let pose = sim.client.get_base_position_and_orientation(body)?;
    println!("Sphere height after 120 steps: {:.3}", pose.translation.z);
    sim.shutdown();
    Ok(())
}
```

Run it without exporting bundled example assets:

```sh
# Linux/macOS
BULLET_SKIP_ASSET_EXPORT=1 cargo +nightly run
```

```powershell
# Windows PowerShell
$env:BULLET_SKIP_ASSET_EXPORT = "1"
cargo +nightly run
```

The program prints a height below the initial `2.0` and exits. It creates its collision geometry in code, so it needs no URDF, meshes, floor, Python, or GUI. The example uses meters, kilograms, seconds, and downward gravity along negative Z. Keep your model scale, masses, gravity, and timestep consistent; these values are passed to Bullet rather than automatically converted.

## Loading your own robot

Supply a URDF and every mesh it references. Implement `robot_behavior::RobotDescription` with `URDF = Some("your_robot.urdf")`, add the model's directory with `sim.add_search_path(...)`, then call `sim.robot_builder::<YourRobot>("robot").base_fixed(true).load()?`. Import the `AddSearchPath`, `AddRobot`, and `EntityBuilder` traits for these methods.

The builder returns `RsBulletRobot<YourRobot>`. Its `body_id`, `joint_indices`, and `joint_names` connect the typed handle to Bullet. The joint list selects revolute and prismatic joints; extra behavior traits on your robot description are needed for the corresponding typed motion APIs. For lower-level model loading or per-joint commands, use `PhysicsClient` directly. Poses use `nalgebra::Isometry3<f64>`; distinguish world/link poses from a joint's relative coordinate when interpreting state.

The crates.io package includes the native engine source, not the full collection of demonstration models. Some repository examples refer to local asset paths and must be adapted. To obtain Bullet's example models explicitly:

```sh
git clone --depth 1 https://github.com/bulletphysics/bullet3.git
```

Point `set_additional_search_path` at that checkout's `examples/pybullet/gym/pybullet_data` directory and keep its meshes beside the URDFs. For a reproducible experiment, pin the model repository revision too. `BULLET_SKIP_ASSET_EXPORT=1` disables build-time asset copying; without it, the build may populate the user's Bullet data directory when that directory does not already exist.

## Optional integration and source builds

The default feature set is empty. Enable `rsbullet = { version = "0.4.0", features = ["roplat"] }` to expose `SimRhythm`; ordinary simulator use does not need roplat. Tokio remains a normal dependency because motion futures use its timers. The backend/transport flags such as `dart`, `physx`, `mujoco`, and `grpc` do not by themselves install or validate those external backends.

For a source checkout, initialize the native source before building:

```sh
git clone https://github.com/Robot-Exp-Platform/rsbullet.git
cd rsbullet
git submodule update --init --recursive rsbullet-sys/bullet3
```

Repository manifests retain pinned Git dependencies; building that source requires access to the declared Git sources. The crates.io release resolves its dependencies through the registry instead. Prefer the registry dependency above for a first application.

## Next steps

- [Public Rust API](https://docs.rs/rsbullet/0.4.0/rsbullet/) and [core API](https://docs.rs/rsbullet-core/0.4.0/rsbullet_core/).
- [API inventory](RSBULLET_API_REFERENCE.md) for the command surface; check current signatures when adapting older PyBullet/RuBullet examples.
- [Examples](rsbullet/examples) for constraints, dynamics, and rendering. Many open a GUI or need external assets.
- [Robot builders and queued control](rsbullet/src/rsbullet_robot.rs), [simulation stepping](rsbullet/src/rsbullet.rs), and [SimRhythm](rsbullet/src/sim_rhythm.rs).
- [roplat_rerun](https://github.com/Robot-Exp-Platform/roplat_rerun) for visualizing simulation state in Rerun.

## License

The Rust crates are distributed under the [MIT license](LICENSE). The vendored Bullet source retains its own upstream license; see [Bullet's license](https://github.com/bulletphysics/bullet3/blob/master/LICENSE.txt). Imported robot models may have separate licenses.
