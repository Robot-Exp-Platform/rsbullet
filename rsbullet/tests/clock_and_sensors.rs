//! Readings and the simulation clock agree: a reading refreshed during a step
//! carries that step's time, and the clock has reached that time once the
//! step returns, so a node woken by the clock reads the refreshed sample.

use std::time::Duration;

use roplat_behavior::{
    AddRobot, EntityBuilder, ForceTorqueSensor, ImuSensor, PhysicsEngine, RobotDescription,
    msgs::{Clock, FrameId},
};
use rsbullet::{Mode, RsBullet};

struct HangingLoad;
impl RobotDescription for HangingLoad {
    const URDF: Option<&'static str> = Some(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/assets/hanging_load.urdf"
    ));
}

#[test]
fn readings_carry_the_time_the_clock_reaches_after_the_step() {
    let mut sim = RsBullet::new(Mode::Direct).unwrap();
    sim.set_gravity([0.0, 0.0, -9.81]).unwrap();
    sim.set_step_time(Duration::from_millis(1)).unwrap();
    let robot = sim
        .robot_builder::<HangingLoad>("load")
        .base_fixed(true)
        .load()
        .unwrap();
    let mut wrench = sim
        .add_force_torque_sensor(&robot, 0, FrameId::new(7))
        .unwrap();
    let mut imu = sim.add_imu(&robot, 0, FrameId::new(3)).unwrap();
    let clock = sim.clock();

    for step in 1..=3 {
        sim.step().unwrap();
        let now = clock.now();
        assert_eq!(now.nanos, step * 1_000_000);
        assert_eq!(now, sim.simulated_stamp());
        assert_eq!(wrench.read_wrench().unwrap().header.stamp, now);
        assert_eq!(imu.read_imu().unwrap().header.stamp, now);
    }
}
