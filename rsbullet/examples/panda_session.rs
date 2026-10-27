//! A Panda driven by the same real-time control graph a real arm would run.
//!
//! cargo run -p rsbullet --no-default-features --features roplat --example panda_session --
//!   /path/to/bullet3/examples/pybullet/gym/pybullet_data
//!
//! `track` is written against `ControlWith<JointPositionControl<7>>` only; here
//! it drives an `RsBulletDevice` under gravity, one integration per cycle.
use std::time::{Duration, Instant};

use roplat::RoplatResult;
use roplat_behavior::behavior::{EndPoint, Joints};
use roplat_behavior::msgs::Clock;
use roplat_behavior::roplat::ControlRhythm;
use roplat_behavior::{
    AddRobot, ControlWith, EntityBuilder, JointPositionControl, JointState, PhysicsEngine, Robot,
    RobotDescription,
};
use rsbullet::{Mode, RsBullet, RsBulletDevice};

/// Joint limits from the Franka Emika Panda datasheet.
struct Panda;

impl RobotDescription for Panda {
    const URDF: Option<&'static str> = Some("franka_panda/panda.urdf");
}

impl Joints<7> for Panda {
    const JOINT_MIN: [f64; 7] = [
        -2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973,
    ];
    const JOINT_MAX: [f64; 7] = [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973];
    const JOINT_VEL_BOUND: [f64; 7] = [2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61];
    const TORQUE_BOUND: [f64; 7] = [87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0];
}

impl EndPoint for Panda {}

const HOME: [f64; 7] = [0.0, -0.6, 0.0, -2.1, 0.0, 1.6, 0.8];
const TARGET: [f64; 7] = [0.4, -0.3, -0.3, -1.8, 0.3, 1.9, 0.5];

/// Minimum-jerk motion from the first observed position to `target` over
/// `duration`, then `settle` more of holding it.
#[roplat::system]
async fn track<R>(
    mut arm: R,
    target: [f64; 7],
    duration: Duration,
    settle: Duration,
) -> RoplatResult<R>
where
    R: ControlWith<JointPositionControl<7>> + Send,
{
    let mut control = ControlRhythm::<R, JointPositionControl<7>>::new();
    let mut start = None;
    let mut elapsed = Duration::ZERO;
    let mut reference = roplat::node::map(move |(state, dt): (JointState<7>, Duration)| {
        let start = *start.get_or_insert_with(|| state.meas.q().unwrap_or(target));
        elapsed += dt;
        let s = (elapsed.as_secs_f64() / duration.as_secs_f64()).min(1.0);
        let blend = s * s * s * (10.0 - 15.0 * s + 6.0 * s * s);
        let command = std::array::from_fn(|i| start[i] + (target[i] - start[i]) * blend);
        (command, elapsed >= duration + settle)
    });
    &mut arm >> control.resources.0;
    control
        >> |sample| {
            sample >> reference;
            reference
        };
    Ok(arm)
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> anyhow::Result<()> {
    let mut args = std::env::args_os().skip(1);
    let assets = args
        .next()
        .ok_or_else(|| anyhow::anyhow!("supply the pinned Bullet model directory"))?;
    anyhow::ensure!(args.next().is_none(), "expected only MODEL_DIRECTORY");

    let mut engine = RsBullet::new(Mode::Direct)?;
    engine.set_gravity([0.0, 0.0, -9.81])?;
    engine
        .client
        .set_additional_search_path(std::path::Path::new(&assets))?;
    let robot = engine
        .robot_builder::<Panda>("panda")
        .joint_names((1..=7).map(|i| format!("panda_joint{i}")))
        .end_effector_link_name("panda_grasptarget")
        .base_fixed(true)
        .load()?;
    for (&joint, &q) in robot.joint_indices.iter().zip(&HOME) {
        engine
            .client
            .reset_joint_state(robot.body_id, joint, q, None)?;
    }
    let clock = engine.clock();
    let started = Instant::now();
    let mut arm = RsBulletDevice::new(engine, robot)?;
    arm = track(
        arm,
        TARGET,
        Duration::from_secs(2),
        Duration::from_millis(500),
    )
    .await?;

    let reached = arm.read_state()?.joint_states;
    let error = reached
        .iter()
        .zip(&TARGET)
        .map(|(joint, target)| (joint.position - target).abs())
        .fold(0.0, f64::max);
    println!(
        "{} integrations, simulated {:?} (clock {:?}), wall {:?}, largest joint error {error:.1e} rad",
        arm.engine().steps_completed(),
        arm.engine().simulated_time(),
        clock.now().as_duration().unwrap_or_default(),
        started.elapsed(),
    );
    anyhow::ensure!(error < 0.02, "the arm did not reach its target");
    Ok(())
}
