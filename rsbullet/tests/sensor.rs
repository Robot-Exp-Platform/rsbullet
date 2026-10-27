//! The simulated force/torque sensor reports the static load it carries.

use std::time::Duration;

use std::f64::consts::FRAC_PI_2;

use nalgebra as na;
use roplat_behavior::{
    AddRobot, EntityBuilder, ForceTorqueSensor, ImuSensor, PhysicsEngine, RobotDescription,
    msgs::{FrameId, Imu, Vector3, clock},
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

fn close(v: Vector3, expected: [f64; 3], tolerance: f64) -> bool {
    (v.x - expected[0]).abs() < tolerance
        && (v.y - expected[1]).abs() < tolerance
        && (v.z - expected[2]).abs() < tolerance
}

#[test]
fn imu_reads_gravity_at_rest_and_nothing_in_free_fall() {
    let mut sim = RsBullet::new(Mode::Direct).unwrap();
    sim.set_gravity([0.0, 0.0, -9.81]).unwrap();
    sim.set_step_time(Duration::from_millis(1)).unwrap();

    // Turned a quarter about x, the body's +y points up.
    let turned = na::Isometry3::rotation(na::Vector3::x() * FRAC_PI_2);
    let fixed = sim
        .robot_builder::<HangingLoad>("fixed")
        .base(turned)
        .base_fixed(true)
        .load()
        .unwrap();
    let mut at_rest = sim.add_imu(&fixed, 0, FrameId::new(3)).unwrap();
    let falling = sim.robot_builder::<HangingLoad>("falling").load().unwrap();
    let mut in_free_fall = sim.add_imu(&falling, -1, FrameId::new(4)).unwrap();
    assert!(sim.add_imu(&fixed, 1, FrameId::new(3)).is_err());

    sim.step().unwrap();
    let first = at_rest.read_imu().unwrap();
    assert_eq!(
        first.value.valid_mask(),
        Imu::ORIENTATION | Imu::ANGULAR_VELOCITY
    );
    for _ in 0..4 {
        sim.step().unwrap();
    }

    let reading = at_rest.read_imu().unwrap();
    assert_eq!(
        (reading.header.seq, reading.header.frame),
        (5, FrameId::new(3))
    );
    assert_eq!(reading.header.stamp, sim.simulated_stamp());
    let q = reading.value.orientation().unwrap();
    let expected = na::UnitQuaternion::from_axis_angle(&na::Vector3::x_axis(), FRAC_PI_2);
    assert!(
        (q.x - expected.i).abs() < 1e-6 && (q.w - expected.w).abs() < 1e-6,
        "{q:?}"
    );
    assert!(close(
        reading.value.angular_velocity().unwrap(),
        [0.0; 3],
        1e-6
    ));
    let a = reading.value.linear_acceleration().unwrap();
    assert!(close(a, [0.0, 9.81, 0.0], 1e-3), "{a:?}");

    // Bullet's default multibody linear damping (0.04) slows the fall a
    // little, so the specific force is small rather than exactly zero.
    let a = in_free_fall
        .read_imu()
        .unwrap()
        .value
        .linear_acceleration()
        .unwrap();
    assert!(close(a, [0.0; 3], 1e-2), "{a:?}");
}

#[test]
fn imu_reads_the_spin_in_the_body_frame() {
    let mut sim = RsBullet::new(Mode::Direct).unwrap();
    sim.set_gravity([0.0, 0.0, 0.0]).unwrap();
    sim.set_step_time(Duration::from_millis(1)).unwrap();
    let robot = sim.robot_builder::<HangingLoad>("spin").load().unwrap();
    let mut imu = sim.add_imu(&robot, -1, FrameId::new(1)).unwrap();
    sim.client
        .reset_base_velocity(robot.body_id, None, Some([0.0, 0.0, 2.0]))
        .unwrap();
    sim.step().unwrap();
    let w = imu.read_imu().unwrap().value.angular_velocity().unwrap();
    assert!(close(w, [0.0, 0.0, 2.0], 1e-2), "{w:?}");
}
