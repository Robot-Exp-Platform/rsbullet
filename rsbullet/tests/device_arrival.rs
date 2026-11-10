//! A joint `MoveTo` of a Panda under gravity, its world stepping at a 4 ms
//! control period, returns with every joint in position, close to the time
//! its S-curve plans, and the arm then holds the target.

use std::time::Duration;

use roplat_behavior::behavior::{EndPoint, JointSpace, Joints};
use roplat_behavior::{Arm, ControlWith, JointLimits, JointPositionControl, MoveTo, Robot};
use roplat_behavior::{RobotDescription, conformance::check_joints};
use roplat_trajectory::path_generate::JointSCurve;
use roplat_world::{AddRobot, EntityBuilder, PhysicsEngine};
use rsbullet::{DeviceModel, Mode, RsBullet, RsBulletDevice};

const PANDA_DATA: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../rsbullet-sys/bullet3/examples/pybullet/gym/pybullet_data"
);

/// A Panda behind a 4 ms control period, the step at which a joint motion
/// used to stop 7 to 11 mrad short of its target.
struct Panda;

impl RobotDescription for Panda {
    const URDF: Option<&'static str> = Some("franka_panda/panda.urdf");
}

impl Joints<7> for Panda {
    const JOINT_LIMITS: JointLimits<7> = JointLimits {
        dq_max: [2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61],
        ddq_max: [15.0, 7.5, 10.0, 12.5, 15.0, 20.0, 20.0],
        dddq_max: [7500.0, 3750.0, 5000.0, 6250.0, 7500.0, 10000.0, 10000.0],
        tau_max: [87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0],
        ..JointLimits::positions(
            [
                -2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973,
            ],
            [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973],
        )
    };
    // The posture the arm stands in; it is not packed for transport here.
    const JOINT_DEFAULT: [f64; 7] = HOME;
    const JOINT_PACKED: [f64; 7] = HOME;
}

impl EndPoint for Panda {}

impl DeviceModel for Panda {
    const CONTROL_PERIOD: Duration = Duration::from_millis(4);
}

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
const TARGET: [f64; 7] = [0.5, -0.2, -0.4, -1.5, 0.4, 2.0, 0.1];

fn device() -> Device {
    let mut world = RsBullet::new(Mode::Direct).unwrap();
    world.set_gravity([0.0, 0.0, -9.81]).unwrap();
    world.set_step_time(Panda::CONTROL_PERIOD).unwrap();
    world.client.set_additional_search_path(PANDA_DATA).unwrap();
    let robot = world
        .robot_builder::<Panda>("panda")
        .joint_names(ARM)
        .base_fixed(true)
        .load()
        .unwrap();
    for (&joint, &q) in robot.joint_indices.iter().zip(&HOME) {
        world
            .client
            .reset_joint_state(robot.body_id, joint, q, None)
            .unwrap();
    }
    let mut device = RsBulletDevice::new(world, robot).unwrap();
    // Stand in the home posture, as an arm does once enabled.
    <Device as ControlWith<JointPositionControl<7>>>::control_with(&mut device, |_, _| {
        (HOME, true)
    })
    .unwrap();
    device
}

fn joints(device: &mut Device) -> [f64; 7] {
    Arm::<7>::state(device).unwrap().joint.meas.q().unwrap()
}

fn worst_error(q: [f64; 7], target: [f64; 7]) -> f64 {
    q.iter()
        .zip(&target)
        .map(|(q, target)| (q - target).abs())
        .fold(0.0, f64::max)
}

#[test]
fn the_device_reports_the_model_period() {
    assert_eq!(<Device as Robot>::CONTROL_PERIOD, Duration::from_millis(4));
    check_joints::<Device, 7>().unwrap();
}

#[test]
fn move_to_returns_in_position_and_the_arm_holds_the_target() {
    let mut device = device();
    let period = device.engine().step_time();
    let start = joints(&mut device);
    let motion = Arm::<7>::motion_limits(&device).joint;
    let planned =
        JointSCurve::new(motion.dq_max, motion.ddq_max, motion.dddq_max).plan(start, TARGET);
    let t0 = device.engine().simulated_time();
    <Device as MoveTo<JointSpace<7>>>::move_to(&mut device, TARGET).unwrap();
    let elapsed = device.engine().simulated_time() - t0;

    // The velocity fed forward keeps the arm within a fraction of a
    // milliradian of the curve (7 to 11 mrad behind it at this step
    // without), so it is in position when the curve ends and the motion
    // returns at the next cycle.
    let arrived = worst_error(joints(&mut device), TARGET);
    assert!(arrived <= 2e-4, "returned {arrived} rad from the target");
    assert!(
        elapsed >= planned && elapsed <= planned + 2 * period,
        "took {elapsed:?}, planned {planned:?}"
    );

    // The target is held, not the position the motion returned at: the
    // motors keep closing on it.
    for _ in 0..(0.5 / period.as_secs_f64()) as usize {
        device.engine_mut().step().unwrap();
    }
    let held = worst_error(joints(&mut device), TARGET);
    assert!(
        held < arrived,
        "holds {held} rad from the target, returned {arrived} rad from it"
    );
}
