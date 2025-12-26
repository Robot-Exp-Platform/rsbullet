use std::time::Duration;

use robot_behavior::ArmState;
use rsbullet_core::{BulletResult, JointState, LinkState, PhysicsClient};

pub(crate) type QueuedControl =
    Box<dyn FnMut(&mut PhysicsClient, Duration) -> BulletResult<bool> + Send + 'static>;

#[derive(Default, Clone)]
pub struct RsBulletRobotState {
    pub joint_states: Vec<JointState>,
    pub link_state: Option<LinkState>,
}

impl<const N: usize> From<RsBulletRobotState> for ArmState<N> {
    fn from(value: RsBulletRobotState) -> Self {
        let mut joint = [0.; N];
        let mut joint_vel = [0.; N];
        let mut torque = [0.; N];

        for i in 0..N {
            joint[i] = value.joint_states[i].position;
            joint_vel[i] = value.joint_states[i].velocity;
            torque[i] = value.joint_states[i].motor_torque;
        }

        let (pose, pose_vel) = if let Some(link_state) = &value.link_state {
            (Some(link_state.world.into()), link_state.world_velocity)
        } else {
            (None, None)
        };

        ArmState {
            joint: Some(joint),
            joint_vel: Some(joint_vel),
            joint_acc: None,
            torque: Some(torque),
            pose_o_to_ee: pose,
            pose_ee_to_k: None,
            cartesian_vel: pose_vel,
            load: None,
        }
    }
}
