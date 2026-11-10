//! Motor forces with and without integration sub-steps: a motor holding a
//! carriage against gravity along its slide reports the carriage's weight,
//! and a force limit below the weight lets it slide. DIRECT engine, no
//! graphics.
use std::time::Duration;

use rsbullet_core::{
    ControlMode, ControlModeArray, Mode, MultiDofControl, PhysicsClient,
    PhysicsEngineParametersUpdate, UrdfOptions,
};

/// The carriage and the sensor link fixed to it (kg) in
/// `tests/assets/linear_axis.urdf`, pulled along the slide by gravity.
const LOAD: f64 = 0.2 + 0.001;
const G: f64 = 9.81;

/// How a motor is told to hold the carriage at 0 with `force` at most.
#[derive(Clone, Copy, Debug)]
enum Command {
    Position,
    Velocity,
    PositionArray,
    PositionMultiDof,
}

/// The carriage's position (m) and the reported motor force (N, from the
/// scalar and the multi-DoF reading) after one second at 240 Hz in
/// `sub_steps` sub-steps (0: Bullet's default, none).
fn hold(
    sub_steps: usize,
    command: Command,
    force: f64,
) -> Result<(f64, f64, f64), Box<dyn std::error::Error>> {
    let mut client = PhysicsClient::connect(Mode::Direct)?;
    client
        .set_time_step(Duration::from_secs_f64(1.0 / 240.0))?
        .set_gravity([-G, 0.0, 0.0])?;
    if sub_steps > 0 {
        client.set_physics_engine_parameter(&PhysicsEngineParametersUpdate {
            num_sub_steps: Some(sub_steps),
            ..Default::default()
        })?;
    }
    let asset = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/assets/linear_axis.urdf");
    let body = client.load_urdf(
        asset,
        Some(UrdfOptions { use_fixed_base: true, ..Default::default() }),
    )?;
    match command {
        Command::Position => {
            client.set_joint_motor_control(body, 0, ControlMode::Position(0.0), Some(force))?
        }
        Command::Velocity => {
            client.set_joint_motor_control(body, 0, ControlMode::Velocity(0.0), Some(force))?
        }
        Command::PositionArray => client.set_joint_motor_control_array(
            body,
            &[0],
            ControlModeArray::Position(&[0.0]),
            Some(&[force]),
        )?,
        Command::PositionMultiDof => client.set_joint_motor_control_multi_dof(
            body,
            0,
            &MultiDofControl::Position {
                target_positions: Some(&[0.0]),
                target_velocities: Some(&[0.0]),
                position_gains: Some(&[0.1]),
                velocity_gains: Some(&[1.0]),
                forces: Some(&[force]),
                damping: None,
                max_velocity: None,
            },
        )?,
    }
    for _ in 0..240 {
        client.step_simulation()?;
    }
    let state = client.get_joint_state(body, 0)?;
    let multi = client.get_joint_state_multi_dof(body, 0)?;
    Ok((state.position, state.motor_torque, multi.motor_torques[0]))
}

#[test]
fn a_holding_motor_reports_the_weight_with_and_without_sub_steps()
-> Result<(), Box<dyn std::error::Error>> {
    let weight = LOAD * G;
    for sub_steps in [0, 1, 2, 4, 8] {
        let (position, force, multi) = hold(sub_steps, Command::Position, 2.0)?;
        assert!(
            position.abs() < 1e-3,
            "{sub_steps} sub-steps: the carriage moved to {position} m"
        );
        for (what, reading) in [("scalar", force), ("multi-DoF", multi)] {
            assert!(
                (reading.abs() - weight).abs() < 0.01 * weight,
                "{sub_steps} sub-steps: the {what} reading is {reading} N for a {weight} N load"
            );
        }
    }
    Ok(())
}

#[test]
fn a_force_limit_below_the_weight_lets_the_load_slide_with_and_without_sub_steps()
-> Result<(), Box<dyn std::error::Error>> {
    for command in [
        Command::Position,
        Command::Velocity,
        Command::PositionArray,
        Command::PositionMultiDof,
    ] {
        for sub_steps in [0, 4] {
            let (position, force, _) = hold(sub_steps, command, 1.0)?;
            // The slide's lower limit is -1 m.
            assert!(
                position < -0.5,
                "{command:?}, {sub_steps} sub-steps: a 1 N motor held the {} N load at {position} m",
                LOAD * G
            );
            // A position motor still pulls towards 0 at its limit; a
            // velocity motor has nothing to do once the slide stops.
            if !matches!(command, Command::Velocity) {
                assert!(
                    (force.abs() - 1.0).abs() < 0.02,
                    "{command:?}, {sub_steps} sub-steps: the motor reports {force} N at its 1 N limit"
                );
            }
        }
    }
    Ok(())
}
