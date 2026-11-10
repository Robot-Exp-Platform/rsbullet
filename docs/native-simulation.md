# Native simulation for robot applications

RsBullet connects Rust directly to Bullet's native C API. Python/PyBullet is not
part of this execution path. Use the [source build instructions](source-build.md)
for the exact Git dependencies, native source submodule and headless features.

## Choose the right layer

| Layer | Responsibility |
| --- | --- |
| `PhysicsClient` | Scene creation, URDF loading, native IK, motor commands, actual state, contacts, cameras |
| `RsBullet` | Ordered queued controls, one native integration, then state/observation refresh |
| `RsBulletRobot<R>` | Selected robot joints, cached actual state, typed queued joint control |
| `RsBulletDevice<R>` | One robot driving its own world: blocking `ControlWith` sessions, joint `MoveTo` and, with a hand, `Gripper` |
| `SimRhythm<E>` | Compatibility alias for `roplat_behavior::roplat::SimulationRhythm<E>`, defaulting to `RsBullet` |
| Application Nodes | Task planning, feedback, commands and evidence processing |

The finite [Panda example](../rsbullet/examples/panda_headless.rs) uses the raw
physics layer, explicit effort caps, actual end-effector readback, and CSV output.
It requires a pinned Bullet model directory and an output path:

```sh
BULLET_SKIP_ASSET_EXPORT=1 ROPLAT_SKIP_ASSET_EXPORT=1 \
  cargo +nightly-2025-11-20 run -p rsbullet --no-default-features \
  --example panda_headless -- \
  rsbullet-sys/bullet3/examples/pybullet/gym/pybullet_data panda-trace.csv
```

It performs 2,400 native integrations and checks two reached poses. This is a
low-level integration example, not a manipulation or perception benchmark.

## Match a device to its backend

`RobotDescription::SIMULATION_MODELS` offers models, optionally restricted to a
backend. `SimulationBackend` selects compatible formats. RsBullet's typed robot
builder declares backend `bullet` and format URDF; its lower-level scene APIs
are separate. Existing `RobotDescription::URDF` remains a fallback.

Selection does not convert models or prove equal actuator, sensor, collision or
dynamic behavior across simulators. A selected model that fails loading produces
an error; it is not silently replaced by a different physical model.

Select the controlled joints explicitly when a device contains several parts:

```rust,ignore
let arm = engine.robot_builder::<Panda>("arm")
    .joint_names([
        "panda_joint1", "panda_joint2", "panda_joint3", "panda_joint4",
        "panda_joint5", "panda_joint6", "panda_joint7",
    ])
    .end_effector_link_name("panda_grasptarget")
    .base_fixed(true)
    .load()?;
```

The two finger joints remain part of the same physical body. Control them through
their native indices with explicit force bounds, or attach them to the device as
its gripper (see below). Requested joint order is
preserved; duplicate, fixed, unknown joints and missing named end effectors are
rejected. A failed builder configuration removes the body it just created.

## Time and state ordering

Set the physics timestep on the engine. `SimRhythm::logical(engine)` adds no
wall-clock delay; `SimRhythm::new(engine, period)` paces tick starts. Wall pacing
does not change physics dt and never skips physics steps to catch up.

The generic scheduler lives in `roplat_behavior` behind its optional `roplat`
feature. Other `PhysicsEngine` backends can use `SimulationRhythm<E>` directly
without a Bullet dependency; RsBullet retains its existing constructor names.

Use `.with_max_steps(n)` for a finite run. The budget is per drive; driving again
continues the same world with a new budget. Zero completes without integration.
The existing unit Yield/Feed contract remains valid. Use `#[roplat::system]` for
application node connections instead of manually calling `Node::process`.

One successful `RsBullet::step()` performs these operations in order:

1. Admit queued controls and execute active controls in submission order.
2. Integrate Bullet once.
3. Refresh loaded robot state and registered `add_step_observer` callbacks.
4. Return to the rhythm, which executes its downstream graph domain.

A graph command queued after observation takes effect at the next integration.
`RsBullet::steps_completed()` and `RsBullet::simulated_time()` describe native integrations. If a
post-step observer fails, the physical integration still counts. The rhythm's
counter counts successful engine step calls, so it may be smaller in that case.

Queued callbacks must not independently integrate or reset the world; those
transitions belong to the engine. Observers receive a mutable native client because Bullet queries require it.
Keep them observational: mutating the world from an observer would change the
meaning of the evidence and is the application's responsibility. The older
control `before`/`after` callbacks surround controller calculation using
pre-integration state; use `add_step_observer` for post-integration evidence.

`LinkState::world` is the inertial center-of-mass pose; `world_link_frame` is the
URDF link origin. Typed arm flange state uses the link origin and transforms COM
linear velocity to that origin. When velocity computation was requested, a
computed zero velocity remains a valid measurement; omitted computation is
`None`. Query AABBs separately; Bullet does not fill them in
its link-state response.

IK targets are always expressed in world coordinates, including when
`current_positions` supplies an initial joint seed. RsBullet accounts for
Bullet's seeded-IK base-COM convention internally. Do not add a compensating
base/inertial offset to the application target.

IK rejects invalid link indices and requests beyond Bullet's command capacity.
Nullspace limits/ranges/rest poses must be supplied together with matching
dimensions. Multi-target IK with nullspace parameters is explicitly unsupported;
the native helper would otherwise silently reduce the request to one target.

## Realtime control sessions

A real arm paces its control session itself: `ControlWith::control_with_flow`
blocks, calling the controller once per device cycle until it reports `done`
or breaks. A robot handle cannot do that, because its world only advances in
`RsBullet::step`. `RsBulletDevice::new(engine, robot)` takes the world together
with one robot loaded into it, so its session can step the world:

1. Observe the robot's native state for the channel.
2. Call the controller with that observation and the physics timestep.
3. Apply the command, then step the engine once (queued controls of other
   bodies run inside that step, as usual).

`Break` applies no command and integrates nothing. `done` applies its command
and integrates once more. However the session ends, including on an error,
the joints are left holding their measured position, as a real arm holds after
its control session; a faulted world is left as it is. The joint position,
velocity, torque and arm-torque channels are provided. The torque channels
command raw joint torque: unlike Franka's FCI, Bullet adds no gravity
compensation. Cartesian `MoveTo` is unsupported.

Sessions keep to the model's declared `JointLimits`. Position commands, of
the joint position channel and of the Cartesian pose channel's inverse
kinematics, are clamped to the position limits, and Bullet's position motors
are given each joint's `dq_max` as the bound of the speed they drive at, so a
far target is approached at that speed rather than at whatever the effort
allows. Velocity commands are clamped to `dq_max`, torque commands to the
efforts. A torque session bounds no speed: Bullet has no per-joint speed bound
for a joint without a motor, and the device does not stop a joint that
exceeds `dq_max` as a real arm's safety controller would.

Joint `MoveTo` returns once the joints are in position, as a real arm's motion
does. It plans an S-curve with the motion limits from the measured position
and streams it to the position motors, each cycle commanding the curve's
position at the start of the cycle with the curve's mean velocity over the
cycle fed forward. Bullet's position motor closes only a tenth of a position
error per step (`kp = 0.1`), so a curve followed by position alone lags by
about nine steps of motion: at a 4 ms step that left joints 7 to 11 mrad
short of the target when the curve ended. Fed forward, the joints stay within
0.15 mrad of the curve (a Panda under gravity at a 4 ms step). After the
curve the target is commanded at rest until every joint is within
`ARRIVAL_WINDOW` (1e-3 rad or m) of it; the joints then hold the target. The
window is not tighter because the position motors do not settle exactly
under a load: that Panda holds its joints up to 9e-5 rad off their targets. Joints that have not arrived `ARRIVAL_TIMEOUT` (1 s of simulated
time) after the curve's end, as against an obstacle, fail the motion with
`RobotException::CommandException` and are left holding where they stopped.

`Robot::CONTROL_PERIOD` of an `RsBulletDevice<R>` is the model's
`DeviceModel::CONTROL_PERIOD`, by default `DEFAULT_CONTROL_PERIOD` (1/240 s,
Bullet's default step); every model type used with a device implements
`DeviceModel`, an empty impl taking the default. A session runs one cycle
per physics step, so the world's step should be set to that period; the
device neither sets nor checks it.

The device holds the world exclusively for the session, as a `ControlRhythm`
holds its robot resource. Use `engine()` / `engine_mut()` between sessions and
`into_parts()` to take the world back. Two robots that must be controlled in
lockstep belong to one session over one combined robot, not to two devices.

The [Panda session example](../rsbullet/examples/panda_session.rs) runs such a
generic graph on a seven-joint Panda under gravity.

The same channels implement `AsyncControlWith`, so an `AsyncControlRhythm`
graph switches too: each callback is awaited, and the session yields to
sibling futures after every integration.

### The hand as a gripper

`device.with_gripper(["panda_finger_joint1", "panda_finger_joint2"])` makes the
device a `roplat_behavior::Gripper`, so code written against a real hand, such
as Franka's, drives the simulated fingers. Like a session, a gripper motion
steps the world and returns when the fingers come to rest:

- The fingers must be prismatic joints outside the controlled joints, closing
  at zero and opening to one shared limit, with finite URDF effort limits. The
  opening is the sum of the two positions.
- A gear constraint (ratio −1, bounded by five times the effort limit) keeps
  the fingers mirrored, as one drive moves both fingers of a real hand and as
  pybullet's Panda example couples them. Without it, two force-limited fingers
  squeezing an object offer no resistance to a push along the closing axis,
  and a held object slides between them when the arm accelerates.
- Both finger targets ramp at half the commanded speed. A move (`force` zero)
  uses the effort limit and fails when it stops more than `MOVE_TOLERANCE`
  (2 mm) short of its width, leaving the fingers holding where they stopped.
- A grasp (`force` positive, at most the effort limit) closes all the way with
  that force, or opens all the way for a width beyond the current opening, so
  an object stops the fingers. It succeeds when they rest within
  `GRASP_EPSILON` (0.01 m, libfranka's default) of the width, then keeps
  squeezing until the next gripper command. A missed grasp is not an error:
  `read_gripper().grasped` is zero, as on the Franka driver. `grasped` stays
  set while the opening remains within the tolerance, so a dropped object
  clears it.
- `home_gripper` opens fully at `HOMING_SPEED`; `stop_gripper` holds the
  measured finger positions and releases a grasp without stepping.
- During a gripper motion the controlled joints keep their last motor commands;
  after a session they hold their measured position, after a joint `MoveTo`
  its target. Arm and
  hand do not move at the same time, because one device owns the world.

Code that takes the arm and the gripper as two separate resources, as with
Franka's separate arm and hand connections, needs both from one device here.

The [planned pick example](../rsbullet/examples/panda_planned_pick.rs) puts the
pieces together on a Panda: RRT-Connect plans each motion against a
`roplat_scene::PlanningScene` built from the same URDF, COPP times it, a
`JointTrajectoryFollower` plays it in a `ControlRhythm` graph on the device,
and the hand grasps and releases through `Gripper`. The scene checks the
arm's self-collision, lets the open fingers reach around the cube, and
carries the cube in the hand once grasped. The example carries the cube over
a wall and checks that it lands within 5 mm of its place. Planning does not
touch Bullet; inverse kinematics comes from `roplat_kinematics`.

`tests/planning_scene.rs` loads the Panda into both and compares link
distances: Bullet never finds two links closer than the scene does, less its
1 mm hull margins, though it often reports them farther apart.

Sessions run in logical time. `.realtime(true)` paces cycle starts at the
physics timestep on the wall clock; late cycles start at once and no
integration is skipped. Only the blocking session can wait; an asynchronous
session with `.realtime(true)` reports unsupported rather than block its
executor. `RsBullet::clock()` is a `roplat_msgs::ManualClock` on
the simulation clock: it reads `simulated_time()` after each completed
integration and its observers, and a reset starts a new clock.

## Simulated cameras

`RsBullet::render_rgbd(&spec)` renders a `CameraSpec` with Bullet's software
renderer (DIRECT works), and `RsBullet::camera(spec)` puts it behind the
`roplat_behavior::Camera` contract. `spec.info()` is the frame's
`CameraInfo`, and the image is the one it describes:

- `Intrinsics { fx, fy, cx, cy, distortion }` are pixels in `CameraInfo`'s
  convention, pixel centres on whole coordinates;
  `Intrinsics::centred(width, height, fov_y)` is a square-pixel pinhole with
  the principal point at the image centre. The renderer samples each pixel
  off its centre, by half a pixel in each axis; the projection is placed to
  make the samples the rays of the pixel centres, to within a few hundredths
  of a pixel (`tests/camera_projection.rs`).
- `Distortion::PlumbBob` and `Distortion::RationalPolynomial` take
  `CameraInfo::d`'s coefficients. A distorted camera renders an undistorted
  image twice as fine that covers every pixel's ray and takes each pixel from
  the rendered pixel nearest its ray: rays land within about a quarter of a
  pixel. The fisheye model is not offered.
- Depth is the `z` of the point a pixel sees, in metres. Infinity is the one
  missing value, as the contract has it: nothing before `far` (a surface at
  the far plane counts as none), and no depth is ever a finite number beyond
  `far`.
- `DepthNoise { sigma, dropout, seed }` adds Gaussian noise with a standard
  deviation of `sigma[0] + sigma[1] z + sigma[2] z²` and loses a share
  `dropout` of the pixels. A pixel's noise is drawn from the seed, the frame's
  simulated time and the pixel alone, so a run reproduces; a reading the
  noise moves outside `near..=far` is missing.

Not modelled: exposure, motion blur, rolling shutter, the occlusion shadows of
stereo and structured-light cameras, multipath and flying pixels at depth
edges.

## Failure, reset and limits

An engine step error stops the rhythm before another downstream tick and retains
the underlying error. Pacing waits respond to cooperative stop; an already
running domain is allowed to return its owned nodes before exit.

A command, integration or observer failure latches the engine as faulted. Reset
or replace the world before stepping again. `RsBullet::reset()` (the
`PhysicsEngine` method) clears controls and observers,
resets counters, and invalidates old robot handles. Fault and engine destruction
also invalidate those handles. `read_state()`, typed `state()` and queued command
submission return errors for invalid handles. The legacy infallible convenience
getters cannot report validity; use the fallible state APIs for evidence.

The public `RsBullet::client` field is a low-level escape hatch. Resetting it
directly, removing managed bodies, integrating independently, or changing its
timestep bypasses the engine's handle validity, cached state and counters. Those
operations are unsupported while using managed robot handles; use engine
lifecycle and timestep methods instead. Standalone `PhysicsClient` applications
own these responsibilities themselves.

High-level joint control intersects declared effort bounds with positive finite
URDF effort limits. If the model provides no limit, supply a usable finite bound
in the robot description. Position/velocity motor commands and explicit torque
commands use these bounds. This is not a claim of physical safety or force
fidelity. Native `PhysicsClient` commands remain caller-controlled: always supply
explicit limits for position/velocity control and explicitly bound torque values.

Current boundaries: generic Cartesian `MoveTo` is unsupported (use native IK);
generic payload mutation reports unsupported; legacy `Arm::with_*` setters do
not configure this simulator. Use robot-description constants, explicit native
commands and actual model dynamics instead. Queued control submission is not a
blocking hardware control session and is not completion of physical motion;
`RsBulletDevice` provides both.

## Preserve useful evidence

Keep source/lock/toolchain/model revisions, initial conditions, physics dt,
solver settings, collision flags, actual per-step state and contacts, commands,
termination outcome, and wall time. Record controller/observation ordering.
Use separate initialization and task phases; do not reset object poses during
a claimed physical grasp. Record unsupported configurations and failed runs.
Finite tests and simulator results do not establish hardware readiness or full
PyBullet API parity.
