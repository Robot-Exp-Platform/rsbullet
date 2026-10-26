# Native Panda pick and place with a roplat graph

`rsbullet/examples/panda_pick_place.rs` is a finite, headless application that picks up a dynamic cube with the Panda's actual finger joints, transports it, releases it, and validates the resulting physical motion. It uses Rust, native Bullet DIRECT, `roplat_behavior`, and a real `#[roplat::system]` graph. It does not require Python, a PyBullet runtime, ROS, or robot hardware.

## Run

From the repository workspace, enable the explicit `roplat` feature and supply the Bullet model-data directory and a fresh output directory:

```sh
cargo run -p rsbullet --no-default-features --features roplat --example panda_pick_place -- \
  a /absolute/path/to/rsbullet-sys/bullet3/examples/pybullet/gym/pybullet_data /absolute/path/to/output/a
cargo run -p rsbullet --no-default-features --features roplat --example panda_pick_place -- \
  b /absolute/path/to/rsbullet-sys/bullet3/examples/pybullet/gym/pybullet_data /absolute/path/to/output/b
cargo run -p rsbullet --no-default-features --features roplat --example panda_pick_place -- \
  a-repeat /absolute/path/to/rsbullet-sys/bullet3/examples/pybullet/gym/pybullet_data /absolute/path/to/output/a-repeat
```

The example rejects an output directory that already exists, preserving earlier evidence. It creates missing parent directories. Use the checked-in lockfile for a reproducible dependency set. The model is the existing `franka_panda/panda.urdf`, selected through `RobotDescription::URDF`. The builder controls `panda_joint1` through `panda_joint7`, resolves `panda_grasptarget` by link name, fixes the base, and discovers the two prismatic finger joints from native joint metadata.

| Configuration | Initial cube center, m | Requested final center, m |
|---|---|---|
| `a`, `a-repeat` | [0.45, -0.08, 0.171] | [0.55, 0.10, 0.170] |
| `b` | [0.50, 0.07, 0.171] | [0.42, -0.12, 0.170] |

The recorded seeds identify these deterministic configurations. This example uses no random generator. `a-repeat` creates an independent native world in a new process with the same physical inputs as `a`.

## Execution and physics contract

`SimRhythm::logical` owns a 5,880-step loop. Its domain connects three nodes: `Sense -> Planner -> Actuate`. The robot handle is borrowed through each node's `Resources` channel. The planner retains stage state across ticks; the System owns the graph topology. The application never drives nodes through a manual `process()` chain.

The engine applies queued commands, integrates once, refreshes robot state, and invokes the world observer. The graph then reads the newly integrated sample and queues one command for the next integration. The native observer bridge stores world evidence; same-cycle planner-to-actuator data follows graph edges. The final rhythm is returned to the caller for explicit engine shutdown.

The scene contains a fixed table with its top at 0.15 m and a 40 mm cube of mass 0.05 kg. The cube's inertial pose is the identity, so its native base pose is also its center. Native mass and friction readback are checked to a tolerance suitable for Bullet's scalar precision. The integration period is 4,166,667 ns, approximately 240 Hz. The finite run therefore represents 24.50000196 simulated seconds. There are 100 solver iterations and no internal substeps.

The stages are settle, approach, descend, close, lift, transport, lower, release, retreat, and final settle. Cartesian trajectories use minimum-jerk interpolation. Native IK receives a world-frame pose and the measured current joint vector. The backend is responsible for converting seeded IK targets into the frame required by native Bullet; the application applies no frame-offset compensation. Both finger motors receive ordinary position commands.

All arm commands have explicit effort caps of [87, 87, 87, 87, 12, 12, 12] Nm; both fingers are capped at 15 N. Cube/table lateral friction is 0.8 and finger friction is 1.0. The cube is placed only by its initial body-creation pose. There are no subsequent cube pose writes, resets, fixed constraints, hidden attachments, or external forces used to fabricate a grasp.

**Collision scope:** the Panda is loaded with explicit flags=0, which disables self-collision. Robot-object, robot-table, and object-table collisions remain enabled. This example makes no claim about Panda self-collision fidelity. It uses the existing collision meshes without redesigning the asset.

## Evidence and acceptance

The executable writes:

- `config.json`: seed/configuration, limits, dimensions, masses, timing and collision flags.
- `world.json`: actual body/joint/link IDs and native mass/friction readback.
- `trace.jsonl`: every integrated step's actual joint positions, velocities and motor efforts; cube pose/quaternion/velocity; end-effector position; individual cube contacts with normal/friction forces and penetration; both finger-force sums; applied target; constraint count; simulation and real time.
- `summary.json`: measured acceptance results and any graph error.
- Ten `STEP-PHASE.ppm` images: 320x240 native TinyRenderer P6 snapshots of the actual scene, including visible table and cube geometry.

A run passes only if it completes 5,880 integrations without a graph/native error, raises the cube center at least 0.10 m, transports it at least 0.12 m while raised and in bilateral finger contact, and records at least 120 raised steps with each finger exerting more than 0.05 N. It must then release the cube within 0.025 m of the requested final center. Over the final 240 steps, neither finger may contact the cube, linear speed must remain below 0.01 m/s, angular speed below 0.1 rad/s, and position drift below 0.002 m. All measured motor efforts must remain within the explicit caps and the native runtime constraint count must remain zero.

Exit code 0 means these criteria passed. Exit code 2 means execution or physics acceptance failed, with all available integrated evidence preserved. Initialization errors return an error before integration. Always retain failed output directories and choose a new directory for each attempt.

## Observed independent application trial, 2026-10-26

The standalone application was compiled against rsbullet 0.6.0 / rsbullet_sys 0.3.3, `robot_behavior` 0.8.0 at `5c10565e40fb760c0dbef2b76d814a9dc6842090`, and roplat 0.5.0 at `d6484b1c595b7d69e8727dc04d471f4bbf57def4`, using nightly-2025-11-20. These are the measured trial inputs, not a promise that later revisions reproduce identical numerical values.

| Run | Maximum lift | Raised transport with bilateral contact | Final goal error | Native run wall time |
|---|---:|---:|---:|---:|
| A | 179.074 mm | 222.991 mm | 19.031 mm | 5.685 s |
| B | 179.041 mm | 224.463 mm | 19.403 mm | 5.225 s |
| Fresh A repeat | 179.074 mm | 222.991 mm | 19.031 mm | 5.019 s |

All three runs passed, including a separate raw-trace verifier. Each had 1,633 raised steps with bilateral finger contact, or 6.804167211 simulated seconds. The approximately 19 mm lateral placement offset arose during finger closing and was retained; it meets the preregistered 25 mm criterion but does not establish millimetric placement accuracy. Individual normal contact forces peaked around 27.8 N and maximum contact penetration was around 1.2 mm. Motor effort caps limit actuator commands; contact reaction forces are not bounded to the same values.

The initial application attempt stopped before integration because it used an inappropriate 1e-12 mass readback tolerance. Native readback was 0.05000000074505806 kg and friction 0.800000011920929; the application now uses 1e-6 tolerance and records those observations. An earlier dependency-resolution attempt was stopped after selecting newer registry versions; the validated workspace lockfile was used for subsequent builds. A native seeded-IK frame bug was fixed and regression-tested in the backend before the accepted physical trials. No application offset workaround was added.

The repository example adds an existing-output-directory guard and a corrected executable usage label to the accepted standalone application source. These are evidence-file handling changes; its physics and controller are unchanged. The original accepted standalone source, binary hash, and run directories remain preserved separately.

This is a bounded application validation of this model, scene, and controller. It is not hardware validation, a self-collision validation, a broad claim about all grasp configurations, or a comparative agent-efficiency benchmark.
