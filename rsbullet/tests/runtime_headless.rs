//! Finite native DIRECT checks; no downloaded assets, GUI, Python, or real devices.
use std::sync::{Arc, Mutex};
use std::time::Duration;

use roplat_behavior::{
    JointVelocityControl, Joints, ModelFormat, Robot, RobotDescription, SimulationModel,
    TorqueControl,
};
use roplat_world::{AddRobot, EntityBuilder, PhysicsEngine};
use rsbullet::{BulletError, Mode, RsBullet};

struct Axis;
impl RobotDescription for Axis {
    const URDF: Option<&'static str> = Some(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/assets/single_axis.urdf"
    ));
}
impl Joints<1> for Axis {
    const JOINT_MIN: [f64; 1] = [-1.0];
    const JOINT_MAX: [f64; 1] = [1.0];
    const JOINT_VEL_BOUND: [f64; 1] = [0.5];
    const TORQUE_BOUND: [f64; 1] = [0.2];
}

struct UrdfEffort;
impl RobotDescription for UrdfEffort {
    const URDF: Option<&'static str> = Axis::URDF;
}
impl Joints<1> for UrdfEffort {
    const JOINT_MIN: [f64; 1] = [-1.0];
    const JOINT_MAX: [f64; 1] = [1.0];
    const JOINT_VEL_BOUND: [f64; 1] = [0.5];
    // TORQUE_BOUND intentionally uses roplat_behavior's unspecified default.
}

#[test]
fn urdf_effort_caps_unspecified_description_limit_in_velocity_and_torque_modes()
-> anyhow::Result<()> {
    let mut engine = RsBullet::new(Mode::Direct)?;
    engine
        .set_step_time(Duration::from_millis(5))?
        .set_gravity([0.0; 3])?;
    let mut robot = engine
        .robot_builder::<UrdfEffort>("axis")
        .base_fixed(true)
        .load()?;
    robot.control_with::<JointVelocityControl<1>, _>(|_, _| ([0.5], false))?;
    let mut maximum_force = 0.0_f64;
    for _ in 0..10 {
        engine.step()?;
        maximum_force = maximum_force.max(robot.read_state()?.joint_states[0].motor_torque.abs());
    }
    assert!(maximum_force > 1.9 && maximum_force <= 2.000_001);

    engine.reset()?;
    engine.set_gravity([0.0; 3])?;
    let mut robot = engine
        .robot_builder::<UrdfEffort>("axis")
        .base_fixed(true)
        .load()?;
    robot.control_with::<TorqueControl<1>, _>(|_, _| ([100.0], false))?;
    engine.step()?;
    // Unit mass, 2 N URDF bound, 5 ms step: velocity must be about 0.01 m/s,
    // rather than the 0.5 m/s an uncapped 100 N command would produce.
    let velocity = robot.read_state()?.joint_states[0].velocity;
    assert!(
        (velocity - 0.01).abs() < 0.000_1,
        "actual velocity={velocity}"
    );
    Ok(())
}

#[test]
fn typed_control_has_bounded_effort_and_fresh_post_step_state() -> anyhow::Result<()> {
    let mut engine = RsBullet::new(Mode::Direct)?;
    let dt = Duration::from_millis(5);
    engine.set_step_time(dt)?.set_gravity([0.0; 3])?;
    let mut robot = engine
        .robot_builder::<Axis>("axis")
        .base_fixed(true)
        .joint_names(["axis"])
        .end_effector_link_name("carriage")
        .load()?;
    assert_eq!(robot.joint_indices, [0]);
    assert_eq!(robot.joint_names, ["axis"]);
    let native = Arc::new(Mutex::new(Vec::new()));
    let captured = native.clone();
    let body = robot.body_id;
    engine.add_step_observer(move |client| {
        captured
            .lock()
            .unwrap()
            .push(client.get_joint_state(body, 0)?);
        Ok(())
    });
    robot.control_with::<JointVelocityControl<1>, _>(|_, _| ([0.5], false))?;

    for expected in 1..=100 {
        engine.step()?;
        let cached = robot.read_state()?;
        let direct = engine.client.get_joint_state(body, 0)?;
        assert_eq!(cached.joint_states[0].position, direct.position);
        assert_eq!(cached.joint_states[0].velocity, direct.velocity);
        assert_eq!(engine.steps_completed(), expected);
        assert!(
            direct.motor_torque.abs() <= 0.200_001,
            "{}",
            direct.motor_torque
        );
    }
    let samples = native.lock().unwrap();
    assert_eq!(samples.len(), 100);
    assert!(samples.last().unwrap().position > 0.01);
    assert!(samples.last().unwrap().position < 0.1);
    assert_eq!(engine.simulated_time(), Duration::from_millis(500));
    assert!(!engine.is_faulted());
    Ok(())
}

#[test]
fn missing_effort_bounds_are_explicit_errors_before_integration() -> anyhow::Result<()> {
    struct Unbounded;
    impl RobotDescription for Unbounded {
        const URDF: Option<&'static str> = Some(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/assets/unspecified_effort.urdf"
        ));
    }
    impl Joints<1> for Unbounded {
        const JOINT_MIN: [f64; 1] = [-1.0];
        const JOINT_MAX: [f64; 1] = [1.0];
    }
    let mut engine = RsBullet::new(Mode::Direct)?;
    let robot = engine
        .robot_builder::<Unbounded>("unbounded")
        .base_fixed(true)
        .load()?;
    robot.control_with::<JointVelocityControl<1>, _>(|_, _| ([0.5], false))?;
    let error = engine.step().unwrap_err();
    assert!(error.to_string().contains("TORQUE_BOUND"));
    assert_eq!(engine.steps_completed(), 0);
    assert!(engine.is_faulted());
    Ok(())
}

#[test]
fn failed_control_does_not_integrate_and_reset_invalidates_old_handles() -> anyhow::Result<()> {
    let mut engine = RsBullet::new(Mode::Direct)?;
    let mut robot = engine
        .robot_builder::<Axis>("axis")
        .base_fixed(true)
        .load()?;
    robot.enqueue(|_, _| {
        Err(BulletError::CommandFailed { message: "injected control failure", code: 42 })
    })?;
    assert!(
        engine
            .step()
            .unwrap_err()
            .to_string()
            .contains("injected control failure")
    );
    assert_eq!(engine.steps_completed(), 0);
    assert!(engine.is_faulted());
    assert!(!robot.is_valid());
    assert!(robot.read_state().is_err());
    assert!(engine.step().unwrap_err().to_string().contains("reset"));
    assert_eq!(engine.steps_completed(), 0);
    engine.reset()?;
    assert!(!engine.is_faulted());
    assert!(robot.read_state().is_err());
    assert!(robot.enqueue(|_, _| Ok(true)).is_err());
    engine.step()?;
    assert_eq!(engine.steps_completed(), 1);
    Ok(())
}

#[test]
fn dropping_engine_invalidates_robot_state_and_commands() -> anyhow::Result<()> {
    let mut engine = RsBullet::new(Mode::Direct)?;
    let mut robot = engine
        .robot_builder::<Axis>("axis")
        .base_fixed(true)
        .load()?;
    assert!(robot.read_state().is_ok());
    drop(engine);
    assert!(!robot.is_valid());
    assert!(robot.read_state().is_err());
    assert!(robot.enqueue(|_, _| Ok(true)).is_err());
    Ok(())
}

#[test]
fn observer_failure_counts_completed_integration_and_requires_reset() -> anyhow::Result<()> {
    let mut engine = RsBullet::new(Mode::Direct)?;
    engine.add_step_observer(|_| {
        Err(BulletError::CommandFailed { message: "injected observer failure", code: 43 })
    });
    assert!(
        engine
            .step()
            .unwrap_err()
            .to_string()
            .contains("injected observer failure")
    );
    assert_eq!(engine.steps_completed(), 1);
    assert!(engine.step().is_err());
    assert_eq!(engine.steps_completed(), 1);
    engine.reset()?;
    assert_eq!(engine.steps_completed(), 0);
    assert_eq!(engine.simulated_time(), Duration::ZERO);
    engine.step()?;
    assert_eq!(engine.steps_completed(), 1);
    Ok(())
}

#[test]
fn completing_a_control_does_not_reorder_other_controls() -> anyhow::Result<()> {
    let mut engine = RsBullet::new(Mode::Direct)?;
    let robot = engine
        .robot_builder::<Axis>("axis")
        .base_fixed(true)
        .load()?;
    let calls = Arc::new(Mutex::new(Vec::new()));
    for label in ['a', 'b', 'c'] {
        let calls = calls.clone();
        robot.enqueue(move |_, _| {
            calls.lock().unwrap().push(label);
            Ok(label == 'a')
        })?;
    }
    engine.step()?;
    engine.step()?;
    assert_eq!(*calls.lock().unwrap(), ['a', 'b', 'c', 'b', 'c']);
    Ok(())
}

#[test]
fn invalid_model_or_joint_selection_fails_without_leaking_a_body() -> anyhow::Result<()> {
    struct MjcfOnly;
    impl RobotDescription for MjcfOnly {
        const SIMULATION_MODELS: &'static [SimulationModel] =
            &[SimulationModel::new(ModelFormat::Mjcf, "unused.xml").for_backend("mujoco")];
    }
    let mut engine = RsBullet::new(Mode::Direct)?;
    let result = engine.robot_builder::<MjcfOnly>("unsupported").load();
    assert!(result.err().unwrap().to_string().contains("bullet"));
    assert_eq!(engine.client.get_num_bodies(), 0);
    assert!(
        engine
            .robot_builder::<Axis>("duplicate")
            .joint_indices(vec![0, 0])
            .load()
            .is_err()
    );
    assert_eq!(engine.client.get_num_bodies(), 0);
    assert!(
        engine
            .robot_builder::<Axis>("missing")
            .joint_names(["absent"])
            .load()
            .is_err()
    );
    assert_eq!(engine.client.get_num_bodies(), 0);
    assert!(
        engine
            .robot_builder::<Axis>("bad-eef")
            .end_effector_link_name("absent")
            .load()
            .is_err()
    );
    assert_eq!(engine.client.get_num_bodies(), 0);
    assert!(engine.set_step_time(Duration::ZERO).is_err());
    Ok(())
}

#[test]
fn incomplete_state_has_no_invented_joint_measurements() {
    let state: roplat_behavior::ArmState<7> = rsbullet::types::RsBulletRobotState::default().into();
    assert!(state.joint.meas.q().is_none());
    assert!(state.joint.meas.dq().is_none());
    assert!(state.joint.meas.tau().is_none());
}

#[test]
fn flange_pose_and_velocity_use_link_origin_instead_of_center_of_mass() {
    let link_state = rsbullet::LinkState {
        world: nalgebra::Isometry3::translation(1.0, 0.0, 0.0),
        local_inertial: nalgebra::Isometry3::translation(1.0, 0.0, 0.0),
        world_link_frame: nalgebra::Isometry3::identity(),
        world_velocity: Some([0.0, 2.0, 0.0, 0.0, 0.0, 2.0]),
        world_aabb: None,
    };
    let state: roplat_behavior::ArmState<1> = rsbullet::types::RsBulletRobotState {
        joint_states: Vec::new(),
        link_state: Some(link_state),
    }
    .into();
    assert_eq!(state.flange.meas.pose().unwrap().position(), [0.0; 3]);
    assert_eq!(
        <[f64; 6]>::from(state.flange.meas.twist().unwrap()),
        [0.0, 0.0, 0.0, 0.0, 0.0, 2.0]
    );
}
