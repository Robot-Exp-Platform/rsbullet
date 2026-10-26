use std::time::Duration;

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use robot_behavior::{ArmState, JointSample, SpatialSample, StateView};
use rsbullet_core::{BulletResult, JointState, LinkState, PhysicsClient};

pub(crate) type QueuedControl =
    Box<dyn FnMut(&mut PhysicsClient, Duration) -> BulletResult<bool> + Send + 'static>;

pub(crate) type StateRefresh =
    Box<dyn FnMut(&mut PhysicsClient) -> BulletResult<()> + Send + 'static>;

pub(crate) struct WorldValidity(pub Arc<AtomicBool>);

impl WorldValidity {
    pub fn new() -> Self {
        Self(Arc::new(AtomicBool::new(true)))
    }

    pub fn invalidate(&self) {
        self.0.store(false, Ordering::Release);
    }
}

impl Drop for WorldValidity {
    fn drop(&mut self) {
        self.invalidate();
    }
}

#[derive(Default, Clone)]
pub struct RsBulletRobotState {
    pub joint_states: Vec<JointState>,
    pub link_state: Option<LinkState>,
}

impl<const N: usize> From<RsBulletRobotState> for ArmState<N> {
    fn from(value: RsBulletRobotState) -> Self {
        // An incomplete native sample is unavailable, never invented zero joints.
        let complete = value.joint_states.len() >= N;
        let joint = complete.then(|| std::array::from_fn(|i| value.joint_states[i].position));
        let joint_vel = complete.then(|| std::array::from_fn(|i| value.joint_states[i].velocity));
        let torque = complete.then(|| std::array::from_fn(|i| value.joint_states[i].motor_torque));

        let (pose, pose_vel) = if let Some(link_state) = &value.link_state {
            // Bullet's `world` and linear velocity refer to the center of mass;
            // the selected flange is the URDF link frame, which may be offset.
            let velocity = link_state.world_velocity.map(|sample| {
                let offset = link_state.world_link_frame.translation.vector
                    - link_state.world.translation.vector;
                let angular = nalgebra::Vector3::new(sample[3], sample[4], sample[5]);
                let linear = nalgebra::Vector3::new(sample[0], sample[1], sample[2])
                    + angular.cross(&offset);
                [
                    linear.x, linear.y, linear.z, angular.x, angular.y, angular.z,
                ]
            });
            (Some(link_state.world_link_frame.into()), velocity)
        } else {
            (None, None)
        };

        ArmState {
            joint: StateView::from_meas(JointSample {
                q: joint,
                dq: joint_vel,
                ddq: None,
                tau: torque,
                dtau: None,
            }),
            flange: StateView::from_meas(SpatialSample {
                pose,
                vel: pose_vel,
                acc: None,
                wrench: None,
            }),
            load: None,
            ..Default::default()
        }
    }
}
