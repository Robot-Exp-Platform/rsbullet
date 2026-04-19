use std::{thread::sleep, time::Duration};

use nalgebra as na;
use rsbullet::{
    BulletResult, ControlModeArray, InverseKinematicsOptions, JointType, Mode, PhysicsClient,
    UrdfOptions,
};

fn main() -> BulletResult<()> {
    let mut client = PhysicsClient::connect(Mode::Gui)?;
    client
        .set_default_search_path()?
        .set_gravity([0., 0., -9.81])?
        .set_time_step(Duration::from_secs_f64(1.0 / 240.0))?;

    client.load_urdf("plane.urdf", None::<()>)?;

    let robot_id = client.load_urdf(
        "franka_panda/panda.urdf",
        Some(UrdfOptions {
            base: Some(na::Isometry3::translation(0.0, 0.0, 0.0)),
            use_fixed_base: true,
            ..Default::default()
        }),
    )?;

    let mut arm_joint_indices = Vec::new();
    for joint_index in 0..client.get_num_joints(robot_id) {
        let info = client.get_joint_info(robot_id, joint_index)?;
        if info.joint_type == JointType::Revolute {
            arm_joint_indices.push(joint_index);
            if arm_joint_indices.len() == 7 {
                break;
            }
        }
    }

    if arm_joint_indices.len() != 7 {
        return Err(rsbullet::BulletError::CommandFailed {
            message: "failed to find 7 Franka arm revolute joints",
            code: arm_joint_indices.len() as i32,
        });
    }

    let default_arm_pose = [0.0, -0.6, 0.0, -2.1, 0.0, 1.6, 0.8];
    for (i, joint_index) in arm_joint_indices.iter().enumerate() {
        client.reset_joint_state(robot_id, *joint_index, default_arm_pose[i], None)?;
    }

    let end_effector_link_index = 11;
    let target_pose = na::Isometry3::from_parts(
        na::Translation3::new(0.45, 0.00, 0.45),
        na::UnitQuaternion::from_euler_angles(std::f64::consts::PI, 0.0, 0.0),
    );

    let dof = client.compute_dof_count(robot_id) as usize;

    let mut tick = 0usize;
    while client.is_connected() {
        let current_arm_q: Vec<f64> = client
            .get_joint_states(robot_id, &arm_joint_indices)?
            .into_iter()
            .map(|s| s.position)
            .collect();

        let mut lower_full = vec![
            -2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973,
        ];
        let mut upper_full = vec![2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973];
        let mut rest_full = default_arm_pose.to_vec();

        while lower_full.len() < dof {
            lower_full.push(0.0);
            upper_full.push(0.04);
            rest_full.push(0.02);
        }
        let joint_ranges: Vec<f64> = lower_full
            .iter()
            .zip(upper_full.iter())
            .map(|(l, u)| u - l)
            .collect();

        let mut seed = current_arm_q;
        while seed.len() < dof {
            seed.push(0.02);
        }

        let ik_options = InverseKinematicsOptions {
            current_positions: Some(&seed),
            lower_limits: Some(&lower_full),
            upper_limits: Some(&upper_full),
            joint_ranges: Some(&joint_ranges),
            rest_poses: Some(&rest_full),
            max_iterations: Some(200),
            residual_threshold: Some(1e-4),
            ..Default::default()
        };

        let solution = client.calculate_inverse_kinematics(
            robot_id,
            end_effector_link_index,
            target_pose,
            &ik_options,
        )?;

        let target_positions: Vec<f64> = solution[..7].to_vec();
        client.set_joint_motor_control_array(
            robot_id,
            &arm_joint_indices,
            ControlModeArray::Position(&target_positions),
            None,
        )?;

        client.step_simulation()?;
        sleep(Duration::from_secs_f64(1.0 / 240.0));

        tick += 1;
        if tick.is_multiple_of(240) {
            let ee_state = client.get_link_state(robot_id, end_effector_link_index, true, false)?;
            let p = ee_state.world_link_frame.translation.vector;
            let t = target_pose.translation.vector;
            let err = (p - t).norm();
            println!(
                "ee=({:.3}, {:.3}, {:.3}), target=({:.3}, {:.3}, {:.3}), pos_err={:.4}",
                p.x, p.y, p.z, t.x, t.y, t.z, err
            );
        }
    }

    Ok(())
}
