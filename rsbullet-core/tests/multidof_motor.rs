//! Native multi-DOF command receipts and measured motion; no GUI or external assets.
use std::time::Duration;

use rsbullet_core::{
    JointType, Mode, MultiDofControl, MultiDofControlEntry, PhysicsClient, UrdfOptions,
};

fn world(asset: &str) -> Result<(PhysicsClient, i32), Box<dyn std::error::Error>> {
    let mut client = PhysicsClient::connect(Mode::Direct)?;
    client.set_time_step(Duration::from_secs_f64(1.0 / 240.0))?;
    client.set_gravity([0.0; 3])?;
    let body = client.load_urdf(
        asset,
        Some(UrdfOptions { use_fixed_base: true, ..Default::default() }),
    )?;
    Ok((client, body))
}

fn send(
    client: &mut PhysicsClient,
    body: i32,
    array: bool,
    control: MultiDofControl<'_>,
) -> rsbullet_core::BulletResult<()> {
    if array {
        client.set_joint_motor_control_multi_dof_array(
            body,
            &[MultiDofControlEntry { joint_index: 0, control }],
        )
    } else {
        client.set_joint_motor_control_multi_dof(body, 0, &control)
    }
}

fn position<'a>(quaternion: &'a [f64], forces: &'a [f64]) -> MultiDofControl<'a> {
    MultiDofControl::Position {
        target_positions: Some(quaternion),
        target_velocities: Some(&[0.0; 3]),
        position_gains: Some(&[0.1; 3]),
        velocity_gains: Some(&[1.0; 3]),
        forces: Some(forces),
        damping: None,
        max_velocity: None,
    }
}

#[test]
fn spherical_position_single_and_array_reach_measured_orientation()
-> Result<(), Box<dyn std::error::Error>> {
    for array in [false, true] {
        let (mut client, body) = world(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/assets/spherical_joint.urdf"
        ))?;
        let info = client.get_joint_info(body, 0)?;
        assert_eq!(info.joint_type, JointType::Spherical);
        assert_eq!((info.q_size, info.u_size), (4, 3));
        let unsupported = send(
            &mut client,
            body,
            array,
            MultiDofControl::Velocity {
                target_velocities: Some(&[0.0, 0.0, 0.2]),
                velocity_gains: Some(&[1.0; 3]),
                forces: Some(&[0.2; 3]),
                damping: None,
            },
        )
        .unwrap_err();
        assert!(
            unsupported
                .to_string()
                .contains("Velocity-only motor control supports")
        );
        let angle = if array { -0.4_f64 } else { 0.4_f64 };
        let target = [0.0, 0.0, (angle / 2.0).sin(), (angle / 2.0).cos()];
        send(&mut client, body, array, position(&target, &[0.2; 3]))?;
        for _ in 0..240 {
            client.step_simulation()?;
            let state = client.get_joint_state_multi_dof(body, 0)?;
            assert_eq!((state.positions.len(), state.velocities.len()), (4, 3));
            assert!(
                state
                    .motor_torques
                    .iter()
                    .all(|x| x.is_finite() && x.abs() <= 0.200_001)
            );
        }
        let state = client.get_joint_state_multi_dof(body, 0)?;
        let dot: f64 = state.positions.iter().zip(target).map(|(a, b)| a * b).sum();
        let error = 2.0 * dot.abs().clamp(0.0, 1.0).acos();
        assert!(
            error < 0.005,
            "array={array}, quaternion={:?}, error={error}",
            state.positions
        );
    }
    Ok(())
}

#[test]
fn spherical_torque_single_and_array_change_actual_angular_velocity()
-> Result<(), Box<dyn std::error::Error>> {
    for array in [false, true] {
        let (mut client, body) = world(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/assets/spherical_joint.urdf"
        ))?;
        // Spherical motors are disabled through position mode with zero effort;
        // Bullet's velocity-only motor mode supports scalar joints, not a ball joint.
        send(
            &mut client,
            body,
            array,
            position(&[0.0, 0.0, 0.0, 1.0], &[0.0; 3]),
        )?;
        for _ in 0..120 {
            send(
                &mut client,
                body,
                array,
                MultiDofControl::Torque { forces: Some(&[0.0, 0.0, 0.004]), damping: None },
            )?;
            client.step_simulation()?;
        }
        let state = client.get_joint_state_multi_dof(body, 0)?;
        assert!(
            state
                .positions
                .iter()
                .chain(&state.velocities)
                .all(|x| x.is_finite())
        );
        assert!(
            state.velocities[2] > 0.2 && state.velocities[2] < 1.0,
            "array={array}, measured angular velocity={:?}",
            state.velocities
        );
        assert!(state.velocities[0].abs() < 1e-3 && state.velocities[1].abs() < 1e-3);
    }
    Ok(())
}

#[test]
fn multidof_velocity_api_single_and_array_move_a_supported_scalar_joint()
-> Result<(), Box<dyn std::error::Error>> {
    for array in [false, true] {
        let (mut client, body) = world(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/assets/linear_axis.urdf"
        ))?;
        send(
            &mut client,
            body,
            array,
            MultiDofControl::Velocity {
                target_velocities: Some(&[0.2]),
                velocity_gains: Some(&[1.0]),
                forces: Some(&[0.2]),
                damping: None,
            },
        )?;
        for _ in 0..120 {
            client.step_simulation()?;
            assert!(client.get_joint_state_multi_dof(body, 0)?.motor_torques[0].abs() <= 0.200_001);
        }
        let state = client.get_joint_state_multi_dof(body, 0)?;
        assert!(
            state.positions[0] > 0.04 && state.positions[0] < 0.11,
            "array={array}, measured position={:?}",
            state.positions
        );
        assert!((state.velocities[0] - 0.2).abs() < 0.01);
    }
    Ok(())
}
