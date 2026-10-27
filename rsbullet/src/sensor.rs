//! Simulated sensors behind the shared device contracts.

use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, AtomicI64, Ordering},
};

use nalgebra as na;
use roplat_behavior::{
    ForceTorqueSensor, RobotException, RobotResult,
    msgs::{ForceTorque, FrameId, Header, Stamp, Wrench, clock},
};
use rsbullet_core::{BulletError, BulletResult};

use crate::{RsBullet, RsBulletRobot};

/// A six-axis force/torque sensor at one joint of a loaded robot.
///
/// The reading is the wrench the parent exerts on the joint's child link,
/// at the joint origin and in the child link's URDF frame (the joint frame),
/// as a sensor mounted at the joint would measure it. Bullet reports it at
/// the child's centre of mass in its inertial frame; the sensor shifts it
/// with the child's local inertial frame rather than switching the
/// engine-wide joint feedback mode. It is sampled after every completed step
/// and stamped with simulation time ([`clock::SIMULATION`]) and the frame
/// given at creation. Before the first step there is no reading. A world
/// reset invalidates the sensor.
pub struct RsBulletForceTorque {
    latest: Arc<Mutex<Option<ForceTorque>>>,
    bias: Wrench,
    world_validity: Arc<AtomicBool>,
}

impl RsBullet {
    /// Enables Bullet's force/torque sensor on `joint_index` of `robot` and
    /// samples it after every step.
    pub fn add_force_torque_sensor<R>(
        &mut self,
        robot: &RsBulletRobot<R>,
        joint_index: i32,
        frame: FrameId,
    ) -> BulletResult<RsBulletForceTorque> {
        if !robot.is_valid() || !Arc::ptr_eq(&robot.world_validity, &self.world_validity()) {
            return Err(BulletError::CommandFailed {
                message: "the robot handle does not belong to this world",
                code: -1,
            });
        }
        let body = robot.body_id;
        if !(0..self.client.get_num_joints(body)).contains(&joint_index) {
            return Err(BulletError::CommandFailed {
                message: "force/torque sensor joint index is outside the robot",
                code: -1,
            });
        }
        self.client
            .enable_joint_force_torque_sensor(body, joint_index, true)?;
        let inertial = self.client.get_dynamics_info(body, joint_index)?;
        let [x, y, z] = inertial.local_inertia_position;
        let [qx, qy, qz, qw] = inertial.local_inertia_orientation;
        let offset = na::Vector3::new(x, y, z);
        let rotation = na::UnitQuaternion::from_quaternion(na::Quaternion::new(qw, qx, qy, qz));

        let latest = Arc::new(Mutex::new(None));
        let sink = latest.clone();
        let nanos = self.simulated_nanos();
        let mut seq = 0;
        self.add_step_observer(move |client| {
            let state = client.get_joint_states(body, &[joint_index])?;
            seq += 1;
            let stamp = Stamp::new(clock::SIMULATION, nanos.load(Ordering::Acquire));
            let [fx, fy, fz, tx, ty, tz] = state[0].force_torque;
            let force = rotation * na::Vector3::new(fx, fy, fz);
            let torque = rotation * na::Vector3::new(tx, ty, tz) + offset.cross(&force);
            let reading = ForceTorque::new(
                Header::new(stamp, frame, seq),
                Wrench::from([force.x, force.y, force.z, torque.x, torque.y, torque.z]),
            );
            *sink.lock().unwrap_or_else(|e| e.into_inner()) = Some(reading);
            Ok(())
        });
        Ok(RsBulletForceTorque {
            latest,
            bias: Wrench::default(),
            world_validity: self.world_validity(),
        })
    }

    pub(crate) fn simulated_nanos(&self) -> Arc<AtomicI64> {
        self.sim_nanos.clone()
    }
}

impl RsBulletForceTorque {
    fn sample(&self) -> RobotResult<ForceTorque> {
        if !self.world_validity.load(Ordering::Acquire) {
            return Err(RobotException::CommandException(
                "simulated sensor is invalid after engine fault, reset, or shutdown".into(),
            ));
        }
        self.latest
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .ok_or_else(|| {
                RobotException::CommandException(
                    "no force/torque sample yet; step the simulation first".into(),
                )
            })
    }
}

impl ForceTorqueSensor for RsBulletForceTorque {
    /// The latest sample minus the bias set by [`bias_wrench`](Self::bias_wrench).
    fn read_wrench(&mut self) -> RobotResult<ForceTorque> {
        let mut reading = self.sample()?;
        let (raw, bias) = (reading.value.to_array(), self.bias.to_array());
        reading.value = Wrench::from(std::array::from_fn::<f64, 6, _>(|i| raw[i] - bias[i]));
        Ok(reading)
    }

    /// Takes the latest sample as zero.
    fn bias_wrench(&mut self) -> RobotResult<()> {
        self.bias = self.sample()?.value;
        Ok(())
    }
}
