//! Simulated sensors behind the shared device contracts.

use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, AtomicI64, Ordering},
};

use nalgebra as na;
use roplat_behavior::{
    ForceTorqueSensor, ImuSensor, RobotException, RobotResult,
    msgs::{ForceTorque, FrameId, Header, Imu, Quaternion, Stamp, Stamped, Vector3, Wrench, clock},
};
use rsbullet_core::{BulletError, BulletResult, PhysicsClient};

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
        let inertial = local_inertial(&mut self.client, body, joint_index)?;
        let (offset, rotation) = (inertial.translation.vector, inertial.rotation);

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

    /// Samples an inertial measurement unit at the URDF frame of `link_index`
    /// of `robot` (`-1` for the base) after every step.
    pub fn add_imu<R>(
        &mut self,
        robot: &RsBulletRobot<R>,
        link_index: i32,
        frame: FrameId,
    ) -> BulletResult<RsBulletImu> {
        if !robot.is_valid() || !Arc::ptr_eq(&robot.world_validity, &self.world_validity()) {
            return Err(BulletError::CommandFailed {
                message: "the robot handle does not belong to this world",
                code: -1,
            });
        }
        let body = robot.body_id;
        if !(-1..self.client.get_num_joints(body)).contains(&link_index) {
            return Err(BulletError::CommandFailed {
                message: "IMU link index is outside the robot",
                code: -1,
            });
        }
        let base_inertial = local_inertial(&mut self.client, body, -1)?;

        let latest = Arc::new(Mutex::new(None));
        let sink = latest.clone();
        let nanos = self.simulated_nanos();
        let mut seq = 0;
        let mut previous: Option<(na::Vector3<f64>, i64)> = None;
        self.add_step_observer(move |client| {
            // Link frame, centre of mass and the centre's world velocity.
            let (link, com, velocity) = if link_index == -1 {
                let com = client.get_base_position_and_orientation(body)?;
                (
                    com * base_inertial.inverse(),
                    com,
                    client.get_base_velocity(body)?,
                )
            } else {
                let state = client.get_link_state(body, link_index, true, true)?;
                let velocity = state.world_velocity.ok_or(BulletError::CommandFailed {
                    message: "Bullet returned no link velocity",
                    code: -1,
                })?;
                (state.world_link_frame, state.world, velocity)
            };
            let [vx, vy, vz, wx, wy, wz] = velocity;
            let angular = na::Vector3::new(wx, wy, wz);
            let offset = link.translation.vector - com.translation.vector;
            let linear = na::Vector3::new(vx, vy, vz) + angular.cross(&offset);
            let now = nanos.load(Ordering::Acquire);

            let rotation = link.rotation;
            let to_body = rotation.inverse();
            let mut imu = Imu::EMPTY
                .with_orientation(Quaternion::from_xyzw([
                    rotation.i, rotation.j, rotation.k, rotation.w,
                ]))
                .with_angular_velocity(vector(to_body * angular));
            if let Some((before, then)) = previous
                && now > then
            {
                let gravity = na::Vector3::from(client.get_physics_engine_parameters()?.gravity);
                let acceleration = (linear - before) / ((now - then) as f64 * 1e-9);
                imu = imu.with_linear_acceleration(vector(to_body * (acceleration - gravity)));
            }
            previous = Some((linear, now));

            seq += 1;
            let stamp = Stamp::new(clock::SIMULATION, now);
            *sink.lock().unwrap_or_else(|e| e.into_inner()) =
                Some(Stamped::new(Header::new(stamp, frame, seq), imu));
            Ok(())
        });
        Ok(RsBulletImu {
            latest,
            world_validity: self.world_validity(),
        })
    }

    pub(crate) fn simulated_nanos(&self) -> Arc<AtomicI64> {
        self.sim_nanos.clone()
    }
}

/// An inertial measurement unit at one link of a loaded robot.
///
/// Orientation is the link frame in the world, angular velocity is in the
/// link frame, and linear acceleration is the specific force an
/// accelerometer measures, `Rᵀ (a − g)` with the engine's current gravity:
/// a link at rest reads `+9.81` up, a falling one reads zero. Acceleration is
/// differenced from the link origin's velocity between steps, so the first
/// sample has none (cleared in the `Imu` mask). Stamps, sequence numbers and
/// validity are as for [`RsBulletForceTorque`].
pub struct RsBulletImu {
    latest: Arc<Mutex<Option<Stamped<Imu>>>>,
    world_validity: Arc<AtomicBool>,
}

impl ImuSensor for RsBulletImu {
    fn read_imu(&mut self) -> RobotResult<Stamped<Imu>> {
        latest_sample(&self.latest, &self.world_validity, "IMU")
    }
}

impl RsBulletForceTorque {
    fn sample(&self) -> RobotResult<ForceTorque> {
        latest_sample(&self.latest, &self.world_validity, "force/torque")
    }
}

fn latest_sample<T: Copy>(
    latest: &Mutex<Option<T>>,
    world_validity: &AtomicBool,
    what: &str,
) -> RobotResult<T> {
    if !world_validity.load(Ordering::Acquire) {
        return Err(RobotException::CommandException(
            "simulated sensor is invalid after engine fault, reset, or shutdown".into(),
        ));
    }
    latest
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .ok_or_else(|| {
            RobotException::CommandException(format!(
                "no {what} sample yet; step the simulation first"
            ))
        })
}

fn vector(v: na::Vector3<f64>) -> Vector3 {
    Vector3::new(v.x, v.y, v.z)
}

/// The link's centre-of-mass frame in its URDF frame, as Bullet loaded it.
fn local_inertial(
    client: &mut PhysicsClient,
    body: i32,
    link: i32,
) -> BulletResult<na::Isometry3<f64>> {
    let info = client.get_dynamics_info(body, link)?;
    let [x, y, z] = info.local_inertia_position;
    let [qx, qy, qz, qw] = info.local_inertia_orientation;
    Ok(na::Isometry3::from_parts(
        na::Translation3::new(x, y, z),
        na::UnitQuaternion::from_quaternion(na::Quaternion::new(qw, qx, qy, qz)),
    ))
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
