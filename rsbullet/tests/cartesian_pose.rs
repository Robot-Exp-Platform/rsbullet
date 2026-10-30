//! The Cartesian pose channel on a Panda: `roplat_control`'s
//! `CartesianTrajectoryFollower` plays a straight move of the grasp target,
//! and each cycle Bullet's inverse kinematics turns the commanded pose into
//! position-motor targets.

use std::{ops::ControlFlow, time::Duration};

use nalgebra::{Isometry3, Translation3, UnitQuaternion};
use roplat_behavior::behavior::{EndPoint, Joints};
use roplat_behavior::{ArmState, CartesianPoseControl, ControlWith, RobotDescription};
use roplat_control::{CartesianTrackingError, CartesianTrajectoryFollower, cartesian_line};
use roplat_world::{AddRobot, EntityBuilder, PhysicsEngine};
use rsbullet::{Mode, RsBullet, RsBulletDevice};

const PANDA_DATA: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../rsbullet-sys/bullet3/examples/pybullet/gym/pybullet_data"
);

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
    const JOINT_ACC_BOUND: [f64; 7] = [15.0, 7.5, 10.0, 12.5, 15.0, 20.0, 20.0];
    const JOINT_JERK_BOUND: [f64; 7] = [7500.0, 3750.0, 5000.0, 6250.0, 7500.0, 10000.0, 10000.0];
    const TORQUE_BOUND: [f64; 7] = [87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0];
}

impl EndPoint for Panda {}

type Device = RsBulletDevice<Panda>;

const ARM: [&str; 7] = [
    "panda_joint1",
    "panda_joint2",
    "panda_joint3",
    "panda_joint4",
    "panda_joint5",
    "panda_joint6",
    "panda_joint7",
];
const HOME: [f64; 7] = [0.0, -0.6, 0.0, -2.1, 0.0, 1.6, 0.8];

/// Bullet caches OBJ meshes in one process-wide map without a lock
/// (`gCachedObjResults` in `LoadMeshFromObj.cpp`), so loading the Panda's
/// meshes in two worlds at once can crash; the tests load one at a time.
static LOADING: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn device() -> Device {
    let _turn = LOADING
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let mut world = RsBullet::new(Mode::Direct).unwrap();
    world.set_gravity([0.0, 0.0, -9.81]).unwrap();
    world.client.set_additional_search_path(PANDA_DATA).unwrap();
    let robot = world
        .robot_builder::<Panda>("panda")
        .joint_names(ARM)
        .end_effector_link_name("panda_grasptarget")
        .base_fixed(true)
        .load()
        .unwrap();
    for (&joint, &q) in robot.joint_indices.iter().zip(&HOME) {
        world
            .client
            .reset_joint_state(robot.body_id, joint, q, None)
            .unwrap();
    }
    RsBulletDevice::new(world, robot).unwrap()
}

/// One cycle holding the measured grasp-target pose; returns it.
fn grasp_target(device: &mut Device) -> Isometry3<f64> {
    let mut seen = None;
    <Device as ControlWith<CartesianPoseControl<7>>>::control_with(device, |state, _| {
        let pose = state
            .flange
            .meas
            .pose()
            .expect("the grasp target is measured");
        seen = Some(pose);
        (pose, true)
    })
    .unwrap();
    seen.unwrap().into()
}

/// Plays `follower` to the end; returns the error that ended it, if any,
/// and the measured poses.
fn follow(
    device: &mut Device,
    follower: &mut CartesianTrajectoryFollower<roplat_behavior::msgs::ManualClock, 7>,
) -> (Option<CartesianTrackingError>, Vec<Isometry3<f64>>) {
    let (mut error, mut seen) = (None, Vec::new());
    <Device as ControlWith<CartesianPoseControl<7>>>::control_with_flow(
        device,
        |state: ArmState<7>, dt: Duration| {
            seen.push(state.flange.meas.pose().unwrap().into());
            match follower.step_by(dt, &state) {
                Ok(step) => ControlFlow::Continue(step),
                Err(e) => {
                    error = Some(e);
                    ControlFlow::Break(())
                }
            }
        },
    )
    .unwrap();
    (error, seen)
}

#[test]
fn the_grasp_target_follows_a_straight_line_down_and_across() {
    let mut device = device();
    let from = grasp_target(&mut device);
    // 10 cm down and 10 cm across in 1.5 s, turning 0.3 rad about the
    // vertical on the way.
    let to =
        Translation3::new(0.0, 0.1, -0.1) * from * UnitQuaternion::from_euler_angles(0.0, 0.0, 0.3);
    let period = Duration::from_secs_f64(1.0 / 240.0);
    let waypoints = cartesian_line(&from, &to, Duration::from_millis(1500), period);
    let mut follower = CartesianTrajectoryFollower::<_, 7>::cycle_timed(waypoints)
        .unwrap()
        .with_path_tolerance(0.01, 0.05)
        .with_goal_tolerance(0.002, 0.01, Duration::from_millis(500));
    let (error, seen) = follow(&mut device, &mut follower);
    assert_eq!(error, None);

    let reached = seen.last().unwrap();
    let off = (reached.translation.vector - to.translation.vector).norm();
    assert!(off < 0.002, "ended {off} m from the goal");
    assert!(reached.rotation.angle_to(&to.rotation) < 0.01);
    // Every measured point near the straight segment.
    let (a, b) = (from.translation.vector, to.translation.vector);
    let worst = seen
        .iter()
        .map(|p| {
            let t =
                ((p.translation.vector - a).dot(&(b - a)) / (b - a).norm_squared()).clamp(0.0, 1.0);
            (p.translation.vector - (a + t * (b - a))).norm()
        })
        .fold(0.0, f64::max);
    assert!(worst < 0.005, "strayed {worst} m from the line");
}

#[test]
fn an_unreachable_goal_ends_the_session_through_the_path_tolerance() {
    let mut device = device();
    let from = grasp_target(&mut device);
    // Two metres out: the arm cannot keep up, and the follower refuses.
    let to = Translation3::new(2.0, 0.0, 0.0) * from;
    let waypoints = cartesian_line(
        &from,
        &to,
        Duration::from_secs(2),
        Duration::from_secs_f64(1.0 / 240.0),
    );
    let mut follower = CartesianTrajectoryFollower::<_, 7>::cycle_timed(waypoints)
        .unwrap()
        .with_path_tolerance(0.02, 0.1);
    let (error, _) = follow(&mut device, &mut follower);
    assert!(
        matches!(error, Some(CartesianTrackingError::PathTolerance { .. })),
        "{error:?}"
    );
}
