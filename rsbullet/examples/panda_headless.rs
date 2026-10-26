//! A finite native Panda IK/state-readback application, without Python or a display.
//!
//! cargo run -p rsbullet --no-default-features --example panda_headless --
//!   /path/to/bullet3/examples/pybullet/gym/pybullet_data /path/to/trace.csv
//! This is a low-level physics example; the roplat examples demonstrate graph scheduling.
use std::{
    fs::File,
    io::{BufWriter, Write},
    time::{Duration, Instant},
};

use nalgebra::{Isometry3, Translation3, UnitQuaternion};
use roplat_behavior::PhysicsEngine;
use rsbullet::{
    ControlModeArray, InverseKinematicsOptions, JointType, Mode, RsBullet, UrdfOptions,
};

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args_os().skip(1);
    let assets = args
        .next()
        .ok_or_else(|| anyhow::anyhow!("supply the pinned Bullet model directory"))?;
    let trace_path = args
        .next()
        .ok_or_else(|| anyhow::anyhow!("supply a CSV output path"))?;
    anyhow::ensure!(
        args.next().is_none(),
        "expected only MODEL_DIRECTORY TRACE.csv"
    );
    let mut trace = BufWriter::new(File::create(trace_path)?);
    writeln!(
        trace,
        "step,sim_seconds,target_x,target_y,target_z,actual_x,actual_y,actual_z,position_error,maximum_arm_motor_torque"
    )?;

    let mut sim = RsBullet::new(Mode::Direct)?;
    let dt = Duration::from_secs_f64(1.0 / 240.0);
    sim.set_step_time(dt)?.set_gravity([0.0, 0.0, -9.81])?;
    sim.client
        .set_additional_search_path(std::path::Path::new(&assets))?;
    sim.client.load_urdf("plane.urdf", None::<()>)?;
    let body = sim.client.load_urdf(
        "franka_panda/panda.urdf",
        Some(UrdfOptions {
            use_fixed_base: true,
            ..Default::default()
        }),
    )?;
    let mut arm = Vec::new();
    let mut fingers = Vec::new();
    let mut ee = None;
    for index in 0..sim.client.get_num_joints(body) {
        let joint = sim.client.get_joint_info(body, index)?;
        if joint.joint_type == JointType::Revolute {
            arm.push(index);
        }
        if joint.joint_type == JointType::Prismatic {
            fingers.push(index);
        }
        if joint.link_name == "panda_grasptarget" {
            ee = Some(index);
        }
    }
    anyhow::ensure!(
        arm.len() == 7 && fingers.len() == 2,
        "expected seven arm joints and two fingers"
    );
    let ee = ee.ok_or_else(|| anyhow::anyhow!("missing panda_grasptarget link"))?;
    let home = [0.0, -0.6, 0.0, -2.1, 0.0, 1.6, 0.8];
    for (&joint, &position) in arm.iter().zip(&home) {
        sim.client.reset_joint_state(body, joint, position, None)?;
    }
    for &joint in &fingers {
        sim.client.reset_joint_state(body, joint, 0.03, None)?;
    }
    let arm_forces = [87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0];
    sim.client.set_joint_motor_control_array(
        body,
        &fingers,
        ControlModeArray::Position(&[0.03, 0.03]),
        Some(&[5.0, 5.0]),
    )?;
    let targets = [[0.45, 0.0, 0.45], [0.45, 0.12, 0.38]];
    let mut final_errors = Vec::new();
    let started = Instant::now();
    for target in targets {
        let pose = Isometry3::from_parts(
            Translation3::from(target),
            UnitQuaternion::from_euler_angles(std::f64::consts::PI, 0.0, 0.0),
        );
        let mut final_error = f64::INFINITY;
        for _ in 0..1200 {
            // The native solver consumes the actual current state. Commands are not observations.
            let seed: Vec<f64> = sim
                .client
                .get_joint_states(body, &arm)?
                .iter()
                .map(|s| s.position)
                .chain([0.03, 0.03])
                .collect();
            let solution = sim.client.calculate_inverse_kinematics(
                body,
                ee,
                pose,
                &InverseKinematicsOptions {
                    current_positions: Some(&seed),
                    max_iterations: Some(100),
                    residual_threshold: Some(1e-5),
                    ..Default::default()
                },
            )?;
            anyhow::ensure!(
                solution.len() >= arm.len(),
                "IK returned too few joint positions"
            );
            sim.client.set_joint_motor_control_array(
                body,
                &arm,
                ControlModeArray::Position(&solution[..7]),
                Some(&arm_forces),
            )?;
            sim.step()?;
            let actual = sim
                .client
                .get_link_state(body, ee, true, true)?
                .world_link_frame
                .translation
                .vector;
            final_error = (actual - pose.translation.vector).norm();
            let states = sim.client.get_joint_states(body, &arm)?;
            for (state, limit) in states.iter().zip(arm_forces) {
                anyhow::ensure!(
                    state.motor_torque.is_finite() && state.motor_torque.abs() <= limit + 1e-6,
                    "motor exceeded explicit effort limit"
                );
            }
            let max_torque = states
                .iter()
                .map(|s| s.motor_torque.abs())
                .fold(0.0_f64, f64::max);
            writeln!(
                trace,
                "{},{:.9},{},{},{},{:.9},{:.9},{:.9},{:.9},{:.9}",
                sim.steps_completed(),
                sim.simulated_time().as_secs_f64(),
                target[0],
                target[1],
                target[2],
                actual.x,
                actual.y,
                actual.z,
                final_error,
                max_torque
            )?;
        }
        anyhow::ensure!(
            final_error < 0.02,
            "Panda did not reach target: actual error {final_error} m"
        );
        final_errors.push(final_error);
    }
    trace.flush()?;
    println!(
        "{{\"backend\":\"native-bullet\",\"python_used\":false,\"steps\":{},\"sim_seconds\":{:.9},\"run_wall_seconds\":{:.6},\"final_errors_m\":{:?}}}",
        sim.steps_completed(),
        sim.simulated_time().as_secs_f64(),
        started.elapsed().as_secs_f64(),
        final_errors
    );
    sim.shutdown();
    Ok(())
}
