//! The simulated force/torque sensor reports the static load it carries.

use std::time::Duration;

use roplat_behavior::{
    AddRobot, EntityBuilder, ForceTorqueSensor, PhysicsEngine, RobotDescription,
    msgs::{FrameId, clock},
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
fn force_torque_sensor_reports_the_static_load() {
    let mut sim = RsBullet::new(Mode::Direct).unwrap();
    sim.set_gravity([0.0, 0.0, -9.81]).unwrap();
    sim.set_step_time(Duration::from_millis(1)).unwrap();
    let robot = sim
        .robot_builder::<HangingLoad>("load")
        .base_fixed(true)
        .load()
        .unwrap();
    let mut sensor = sim
        .add_force_torque_sensor(&robot, 0, FrameId::new(7))
        .unwrap();

    assert!(
        sensor.read_wrench().is_err(),
        "no reading before the first step"
    );
    assert!(
        sim.add_force_torque_sensor(&robot, 1, FrameId::new(7))
            .is_err()
    );

    for _ in 0..5 {
        sim.step().unwrap();
    }
    let reading = sensor.read_wrench().unwrap();
    assert_eq!(reading.header.seq, 5);
    assert_eq!(reading.header.frame, FrameId::new(7));
    assert_eq!(reading.header.stamp.clock, clock::SIMULATION);
    assert_eq!(reading.header.stamp.nanos, 5_000_000);
    assert_eq!(sim.simulated_stamp(), reading.header.stamp);

    // The base holds the load up with m g = 19.62 N and against its moment
    // (0.1, 0, 0) × (0, 0, −m g): the reaction torque is −1.962 N·m about y.
    let expected = [0.0, 0.0, 19.62, 0.0, -1.962, 0.0];
    let wrench = reading.value.to_array();
    assert!(
        wrench
            .iter()
            .zip(expected)
            .all(|(a, b)| (a - b).abs() < 1e-4),
        "{wrench:?}"
    );

    sensor.bias_wrench().unwrap();
    sim.step().unwrap();
    let tared = sensor.read_wrench().unwrap().value.to_array();
    assert!(tared.iter().all(|v| v.abs() < 1e-6), "{tared:?}");

    sim.reset().unwrap();
    assert!(
        sensor.read_wrench().is_err(),
        "a reset invalidates the sensor"
    );
}
