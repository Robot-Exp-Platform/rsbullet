#![allow(dead_code)]

use roplat_behavior::behavior::{EndPoint, Joints};
use roplat_behavior::{JointLimits, Robot, RobotDescription};
use roplat_world::{AddRobot, EntityBuilder};
use rsbullet::{Mode, RsBullet, RsBulletDevice, RsBulletRobot};

/// One prismatic axis, effort 100 N and position limits of ±1 m.
pub struct Axis;

impl RobotDescription for Axis {
    const URDF: Option<&'static str> = Some(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/assets/servo_axis.urdf"
    ));
}

impl Joints<1> for Axis {
    const JOINT_LIMITS: JointLimits<1> = JointLimits {
        dq_max: [1.0],
        ddq_max: [4.0],
        dddq_max: [40.0],
        ..JointLimits::positions([-1.0], [1.0])
    };
}

impl EndPoint for Axis {}

pub fn load(engine: &mut RsBullet, y: f64) -> anyhow::Result<RsBulletRobot<Axis>> {
    engine
        .robot_builder::<Axis>("axis")
        .base(nalgebra::Isometry3::translation(0.0, y, 0.0))
        .base_fixed(true)
        .load()
}

pub fn device() -> anyhow::Result<RsBulletDevice<Axis>> {
    let mut engine = RsBullet::new(Mode::Direct)?;
    let robot = load(&mut engine, 0.0)?;
    Ok(RsBulletDevice::new(engine, robot)?)
}

pub fn position<R: Robot<State = rsbullet::types::RsBulletRobotState>>(
    robot: &mut R,
) -> anyhow::Result<f64> {
    Ok(robot.read_state()?.joint_states[0].position)
}
