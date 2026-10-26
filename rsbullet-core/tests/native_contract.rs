//! Real DIRECT-engine checks. No Python, graphics, downloaded model, or hardware.
use std::time::Duration;

use nalgebra::Isometry3;
use rsbullet_core::{
    CameraImageOptions, CollisionGeometry, CollisionId, CollisionShapeOptions, ControlMode,
    ControlModeArray, Mode, MultiBodyBase, MultiBodyCreateOptions, PhysicsClient,
    PhysicsEngineParametersUpdate, Renderer, UrdfOptions, VisualGeometry, VisualId,
};

#[test]
fn native_motion_contact_validation_and_reset() -> Result<(), Box<dyn std::error::Error>> {
    let mut client = PhysicsClient::connect(Mode::Direct)?;
    let dt = Duration::from_secs_f64(1.0 / 240.0);
    client.set_time_step(dt)?.set_gravity([0.0, 0.0, -9.81])?;
    assert!(client.set_time_step(Duration::ZERO).is_err());
    assert!(
        client
            .set_physics_engine_parameter(&PhysicsEngineParametersUpdate {
                fixed_time_step: Some(Duration::ZERO),
                ..Default::default()
            })
            .is_err()
    );
    assert!(client.set_gravity([f64::NAN, 0.0, 0.0]).is_err());

    let ground_shape = client.create_collision_shape(
        &CollisionGeometry::Box {
            half_extents: [1.0, 1.0, 0.05],
        },
        Some(CollisionShapeOptions::default()),
    )?;
    let ground = client.create_multi_body(&MultiBodyCreateOptions {
        base: MultiBodyBase {
            mass: 0.0,
            pose: Isometry3::translation(0.0, 0.0, -0.05),
            collision_shape: CollisionId(ground_shape),
            ..Default::default()
        },
        ..Default::default()
    })?;
    let sphere_shape = client.create_collision_shape(
        &CollisionGeometry::Sphere { radius: 0.05 },
        Some(CollisionShapeOptions::default()),
    )?;
    let sphere = client.create_multi_body(&MultiBodyCreateOptions {
        base: MultiBodyBase {
            mass: 1.0,
            pose: Isometry3::translation(0.0, 0.0, 1.0),
            collision_shape: CollisionId(sphere_shape),
            ..Default::default()
        },
        ..Default::default()
    })?;
    for _ in 0..480 {
        client.step_simulation()?;
    }
    let z = client
        .get_base_position_and_orientation(sphere)?
        .translation
        .z;
    assert!((z - 0.05).abs() < 0.005, "actual sphere z={z}");
    let contacts = client.get_contact_points(Some(sphere), Some(ground), None, None)?;
    assert!(
        !contacts.is_empty(),
        "settled sphere must physically contact ground"
    );
    assert!(
        contacts
            .iter()
            .all(|p| p.normal_force.is_finite() && p.normal_force >= 0.0)
    );
    let support: f64 = contacts.iter().map(|p| p.normal_force).sum();
    assert!(
        (support - 9.81).abs() < 0.5,
        "actual support force={support}"
    );

    client.reset_simulation()?;
    assert_eq!(client.get_num_bodies(), 0);
    client.set_time_step(dt)?.set_gravity([0.0, 0.0, 0.0])?;
    let asset = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/assets/linear_axis.urdf");
    let axis = client.load_urdf(
        asset,
        Some(UrdfOptions {
            use_fixed_base: true,
            ..Default::default()
        }),
    )?;
    assert_eq!(client.get_joint_info(axis, 0)?.joint_name, "slide");
    assert_eq!(client.get_joint_info(axis, 0)?.link_name, "carriage");
    assert!(client.get_link_state(axis, -1, true, true).is_err());
    assert!(client.get_link_states(axis, &[0, 99], true, true).is_err());
    assert_eq!(
        client.get_link_state(axis, 0, true, true)?.world_velocity,
        Some([0.0; 6])
    );
    assert!(
        client
            .get_link_state(axis, 0, true, false)?
            .world_velocity
            .is_none()
    );
    assert!(
        client
            .get_link_states(axis, &[0, 1], true, false)?
            .iter()
            .all(|s| s.world_velocity.is_none())
    );
    assert!(
        client
            .get_link_state(axis, 0, true, true)?
            .world_aabb
            .is_none()
    );
    assert!(
        client
            .set_joint_motor_control(axis, 0, ControlMode::Position(f64::NAN), Some(2.0))
            .is_err()
    );
    assert!(
        client
            .set_joint_motor_control(axis, 0, ControlMode::Position(0.25), Some(-1.0))
            .is_err()
    );
    assert!(
        client
            .set_joint_motor_control_array(axis, &[], ControlModeArray::Position(&[1.0]), None)
            .is_err()
    );
    assert!(
        client
            .set_joint_motor_control_array(
                axis,
                &[0],
                ControlModeArray::Position(&[0.25]),
                Some(&[f64::INFINITY])
            )
            .is_err()
    );
    assert!(
        client
            .set_joint_motor_control_array(
                axis,
                &[1],
                ControlModeArray::Position(&[0.25]),
                Some(&[2.0])
            )
            .is_err()
    );
    assert!(
        client
            .set_joint_motor_control(axis, 1, ControlMode::Velocity(1.0), Some(2.0))
            .is_err()
    );

    client.set_joint_motor_control_array(
        axis,
        &[0],
        ControlModeArray::Position(&[0.25]),
        Some(&[2.0]),
    )?;
    let mut max_motor_force = 0.0_f64;
    for _ in 0..480 {
        client.step_simulation()?;
        let state = client.get_joint_state(axis, 0)?;
        max_motor_force = max_motor_force.max(state.motor_torque.abs());
    }
    let position = client.get_joint_state(axis, 0)?.position;
    assert!(
        (position - 0.25).abs() < 0.001,
        "actual axis position={position}"
    );
    assert!(
        max_motor_force <= 2.0 + 1e-6,
        "actual maximum actuator force={max_motor_force}"
    );
    client.set_joint_motor_control(axis, 0, ControlMode::Position(0.10), Some(2.0))?;
    for _ in 0..240 {
        client.step_simulation()?;
    }
    assert!((client.get_joint_state(axis, 0)?.position - 0.10).abs() < 0.001);
    client.disconnect();
    // A new independent engine must remain usable after explicit teardown.
    let mut next = PhysicsClient::connect(Mode::Direct)?;
    next.step_simulation()?;
    Ok(())
}

#[test]
fn tiny_renderer_produces_real_headless_rgb_depth_and_segmentation()
-> Result<(), Box<dyn std::error::Error>> {
    let mut client = PhysicsClient::connect(Mode::Direct)?;
    let visual = client.create_visual_shape(
        &VisualGeometry::Sphere { radius: 0.05 },
        None::<rsbullet_core::VisualShapeOptions>,
    )?;
    let body = client.create_multi_body(&MultiBodyCreateOptions {
        base: MultiBodyBase {
            mass: 0.0,
            visual_shape: VisualId(visual),
            ..Default::default()
        },
        ..Default::default()
    })?;
    let frame = client.get_camera_image(&CameraImageOptions {
        width: 64,
        height: 64,
        view_matrix: Some(PhysicsClient::compute_view_matrix(
            [0.2, 0.2, 0.2],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        )),
        projection_matrix: Some(PhysicsClient::compute_projection_matrix_fov(
            60.0, 1.0, 0.01, 2.0,
        )),
        renderer: Some(Renderer::TinyRenderer),
        ..Default::default()
    })?;
    assert_eq!(frame.rgba.len(), 64 * 64 * 4);
    assert_eq!(frame.depth.len(), 64 * 64);
    assert_eq!(frame.segmentation_mask.len(), 64 * 64);
    assert!(
        frame.segmentation_mask.contains(&body),
        "render must contain the actual body"
    );
    assert!(
        frame.segmentation_mask.contains(&-1),
        "render must also contain background"
    );
    assert!(frame.depth.iter().all(|value| value.is_finite()));
    assert!(frame.depth.iter().any(|value| *value < 1.0));
    Ok(())
}

#[cfg(not(feature = "gui"))]
#[test]
fn headless_build_rejects_gui_without_consuming_a_gui_slot() {
    assert!(PhysicsClient::connect(Mode::Gui).is_err());
    assert!(PhysicsClient::connect(Mode::Gui).is_err());
    assert!(PhysicsClient::connect(Mode::Direct).is_ok());
}

#[test]
fn seeded_ik_keeps_world_targets_on_translated_rotated_inertial_bases()
-> Result<(), Box<dyn std::error::Error>> {
    use nalgebra::{Translation3, UnitQuaternion};
    use rsbullet_core::{InverseKinematicsMultiTargetOptions, InverseKinematicsOptions};
    let mut client = PhysicsClient::connect(Mode::Direct)?;
    let base = Isometry3::from_parts(
        Translation3::new(0.5, 0.2, 0.3),
        UnitQuaternion::from_euler_angles(0.0, 0.0, std::f64::consts::FRAC_PI_2),
    );
    let body = client.load_urdf(
        concat!(env!("CARGO_MANIFEST_DIR"), "/tests/assets/offset_axis.urdf"),
        Some(UrdfOptions {
            base: Some(base),
            use_fixed_base: true,
            ..Default::default()
        }),
    )?;
    let target = base * Isometry3::translation(0.25, 0.0, 0.0);
    for index in [-1, 1] {
        assert!(
            client
                .calculate_inverse_kinematics(
                    body,
                    index,
                    target,
                    &InverseKinematicsOptions::default(),
                )
                .is_err()
        );
    }
    let mut multi_options = InverseKinematicsMultiTargetOptions {
        end_effector_link_indices: &[0],
        target_positions: &[[0.5, 0.45, 0.3]],
        lower_limits: None,
        upper_limits: None,
        joint_ranges: None,
        rest_poses: None,
        joint_damping: None,
        current_positions: None,
        solver: None,
        max_iterations: None,
        residual_threshold: None,
    };
    for index in [-1, 1] {
        let indices = [index];
        let options = InverseKinematicsMultiTargetOptions {
            end_effector_link_indices: &indices,
            ..multi_options.clone()
        };
        assert!(
            client
                .calculate_inverse_kinematics2(body, &options)
                .is_err()
        );
    }
    let indices = [0; 129];
    let positions = [[0.5, 0.45, 0.3]; 129];
    assert!(
        client
            .calculate_inverse_kinematics2(
                body,
                &InverseKinematicsMultiTargetOptions {
                    end_effector_link_indices: &indices,
                    target_positions: &positions,
                    ..multi_options.clone()
                }
            )
            .is_err()
    );
    multi_options.lower_limits = Some(&[-1.0]);
    assert!(
        client
            .calculate_inverse_kinematics2(body, &multi_options)
            .is_err()
    );
    multi_options.upper_limits = Some(&[1.0]);
    multi_options.joint_ranges = Some(&[2.0]);
    multi_options.rest_poses = Some(&[0.0]);
    multi_options.end_effector_link_indices = &[0, 0];
    multi_options.target_positions = &[[0.5, 0.45, 0.3]; 2];
    let error = client
        .calculate_inverse_kinematics2(body, &multi_options)
        .unwrap_err();
    assert!(error.to_string().contains("only for a single target"));
    for seed in [None, Some(&[0.0][..])] {
        let solution = client.calculate_inverse_kinematics(
            body,
            0,
            target,
            &InverseKinematicsOptions {
                current_positions: seed,
                max_iterations: Some(200),
                residual_threshold: Some(1e-6),
                ..Default::default()
            },
        )?;
        assert!(
            (solution[0] - 0.25).abs() < 1e-4,
            "world-target IK solution={solution:?}, seed={seed:?}"
        );
        let positions = [[
            target.translation.x,
            target.translation.y,
            target.translation.z,
        ]];
        let multi = client.calculate_inverse_kinematics2(
            body,
            &InverseKinematicsMultiTargetOptions {
                end_effector_link_indices: &[0],
                target_positions: &positions,
                lower_limits: None,
                upper_limits: None,
                joint_ranges: None,
                rest_poses: None,
                joint_damping: None,
                current_positions: seed,
                solver: None,
                max_iterations: Some(200),
                residual_threshold: Some(1e-6),
            },
        )?;
        assert!(
            (multi[0] - 0.25).abs() < 1e-4,
            "world-target multi IK solution={multi:?}, seed={seed:?}"
        );
    }
    Ok(())
}
