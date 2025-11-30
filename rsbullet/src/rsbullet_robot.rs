use std::sync::{Arc, Mutex};
use std::time::Duration;
use std::{marker::PhantomData, sync::mpsc::Sender};

use anyhow::Result;
use robot_behavior::behavior::{Arm, ArmParam};
use robot_behavior::utils::{isometry_to_raw_parts, path_generate};
use robot_behavior::{
    ArmPreplannedPath, ArmState, Coord, LoadState, MotionType, Pose, RobotException, RobotFile,
    RobotResult, behavior::*,
};
use rsbullet_core::{
    BulletError, BulletResult, ControlModeArray, InverseKinematicsOptions, LoadModelFlags,
    PhysicsClient, UrdfOptions,
};

use crate::RsBullet;
use crate::types::{QueuedControl, RsBulletRobotState};

pub struct RsBulletRobot<R> {
    pub body_id: i32,
    pub joint_indices: Vec<i32>,
    pub(crate) command_sender: Sender<QueuedControl>,
    pub end_effector_link: i32,
    state_cache: Arc<Mutex<RsBulletRobotState>>,
    _marker: PhantomData<R>,
}

impl<R> RsBulletRobot<R> {
    pub fn enqueue<CF>(&self, control: CF) -> anyhow::Result<()>
    where
        CF: FnMut(&mut PhysicsClient, Duration) -> BulletResult<bool> + Send + 'static,
    {
        self.command_sender
            .send(Box::new(control))
            .map_err(|_| anyhow::anyhow!("Failed to send control command: channel closed"))?;
        Ok(())
    }
}

pub struct RsBulletRobotBuilder<'a, R> {
    pub(crate) _marker: PhantomData<R>,
    pub(crate) rsbullet: &'a mut RsBullet,
    pub(crate) load_file: &'static str,
    pub(crate) base: Option<nalgebra::Isometry3<f64>>,
    pub(crate) base_fixed: bool,
    pub(crate) use_maximal_coordinates: Option<bool>,
    pub(crate) scaling: Option<f64>,
    pub(crate) flags: Option<LoadModelFlags>,
}

impl<'a, R: RobotFile> RsBulletRobotBuilder<'a, R> {
    pub fn new(rsbullet: &'a mut RsBullet) -> Self {
        RsBulletRobotBuilder {
            _marker: PhantomData,
            rsbullet,
            load_file: R::URDF,
            base: None,
            base_fixed: false,
            scaling: None,
            flags: None,
            use_maximal_coordinates: None,
        }
    }
}

impl<R> RsBulletRobotBuilder<'_, R> {
    pub fn use_maximal_coordinates(mut self, use_maximal: bool) -> Self {
        self.use_maximal_coordinates = Some(use_maximal);
        self
    }
    pub fn flags(mut self, flags: LoadModelFlags) -> Self {
        self.flags = Some(flags);
        self
    }
}

impl<'a, R> EntityBuilder<'a> for RsBulletRobotBuilder<'a, R> {
    type Entity = RsBulletRobot<R>;

    fn name(self, _: String) -> Self {
        self
    }

    fn base(mut self, base: impl Into<nalgebra::Isometry3<f64>>) -> Self {
        self.base = Some(base.into());
        self
    }
    fn base_fixed(mut self, base_fixed: bool) -> Self {
        self.base_fixed = base_fixed;
        self
    }
    fn scaling(mut self, scaling: f64) -> Self {
        self.scaling = Some(scaling);
        self
    }
    fn load(self) -> Result<RsBulletRobot<R>> {
        let body_id = self.rsbullet.client_mut().load_urdf(
            self.load_file,
            Some(UrdfOptions {
                base: self.base,
                use_fixed_base: self.base_fixed,
                global_scaling: self.scaling,
                flags: self.flags,
                use_maximal_coordinates: self.use_maximal_coordinates,
            }),
        )?;

        let joint_count = self.rsbullet.client_mut().get_num_joints(body_id);
        let joint_indices: Vec<i32> = (0..joint_count).collect();
        let end_effector_link = if joint_count == 0 {
            -1
        } else {
            joint_count - 1
        };

        let joint_states = self
            .rsbullet
            .client_mut()
            .get_joint_states(body_id, &joint_indices)?;
        let link_state = if end_effector_link >= 0 {
            Some(self.rsbullet.client_mut().get_link_state(
                body_id,
                end_effector_link,
                true,
                true,
            )?)
        } else {
            None
        };
        let state_cache = Arc::new(Mutex::new(RsBulletRobotState { joint_states, link_state }));

        let cache_clone = state_cache.clone();
        let joint_indices_clone = joint_indices.clone();
        let end_effector_link_clone = end_effector_link;

        let sender = self.rsbullet.command_sender();

        let robot = RsBulletRobot::<R> {
            body_id,
            joint_indices,
            command_sender: sender,
            end_effector_link,
            state_cache,
            _marker: PhantomData,
        };

        robot.enqueue(move |client, _| {
            let joint_states = client.get_joint_states(body_id, &joint_indices_clone)?;
            let link_state = if end_effector_link_clone >= 0 {
                Some(client.get_link_state(body_id, end_effector_link_clone, true, true)?)
            } else {
                None
            };
            let mut cache = cache_clone.lock().map_err(|_| BulletError::CommandFailed {
                message: "state cache poisoned",
                code: -1,
            })?;
            cache.joint_states = joint_states;
            cache.link_state = link_state;
            Ok(false)
        })?;

        Ok(robot)
    }
}

impl<const N: usize, R> Arm<N> for RsBulletRobot<R>
where
    R: ArmParam<N>,
{
    fn state(&mut self) -> RobotResult<ArmState<N>> {
        let cache = self
            .state_cache
            .lock()
            .map_err(|_| RobotException::NetworkError("state cache poisoned".to_string()))?;
        Ok(cache.clone().into())
    }

    fn set_load(&mut self, _load: LoadState) -> RobotResult<()> {
        Ok(())
    }

    fn set_coord(&mut self, _coord: Coord) -> RobotResult<()> {
        Ok(())
    }

    fn with_coord(&mut self, _coord: Coord) -> &mut Self {
        self
    }

    fn set_speed(&mut self, _speed: f64) -> RobotResult<()> {
        Ok(())
    }

    fn with_speed(&mut self, _speed: f64) -> &mut Self {
        self
    }

    fn with_velocity(&mut self, _joint_vel: &[f64; N]) -> &mut Self {
        self
    }

    fn with_acceleration(&mut self, _joint_acc: &[f64; N]) -> &mut Self {
        self
    }

    fn with_jerk(&mut self, _joint_jerk: &[f64; N]) -> &mut Self {
        self
    }

    fn with_cartesian_velocity(&mut self, _cartesian_vel: f64) -> &mut Self {
        self
    }

    fn with_cartesian_acceleration(&mut self, _cartesian_acc: f64) -> &mut Self {
        self
    }

    fn with_cartesian_jerk(&mut self, _cartesian_jerk: f64) -> &mut Self {
        self
    }

    fn with_rotation_velocity(&mut self, _rotation_vel: f64) -> &mut Self {
        self
    }

    fn with_rotation_acceleration(&mut self, _rotation_acc: f64) -> &mut Self {
        self
    }

    fn with_rotation_jerk(&mut self, _rotation_jerk: f64) -> &mut Self {
        self
    }
}

impl<const N: usize, R> ArmParam<N> for RsBulletRobot<R>
where
    R: ArmParam<N>,
{
    const JOINT_DEFAULT: [f64; N] = R::JOINT_DEFAULT;
    const JOINT_MIN: [f64; N] = R::JOINT_MIN;
    const JOINT_MAX: [f64; N] = R::JOINT_MAX;
    const JOINT_VEL_BOUND: [f64; N] = R::JOINT_VEL_BOUND;
    const JOINT_ACC_BOUND: [f64; N] = R::JOINT_ACC_BOUND;
    const JOINT_JERK_BOUND: [f64; N] = R::JOINT_JERK_BOUND;
    const CARTESIAN_VEL_BOUND: f64 = R::CARTESIAN_VEL_BOUND;
    const CARTESIAN_ACC_BOUND: f64 = R::CARTESIAN_ACC_BOUND;
    const CARTESIAN_JERK_BOUND: f64 = R::CARTESIAN_JERK_BOUND;
    const ROTATION_VEL_BOUND: f64 = R::ROTATION_VEL_BOUND;
    const ROTATION_ACC_BOUND: f64 = R::ROTATION_ACC_BOUND;
    const ROTATION_JERK_BOUND: f64 = R::ROTATION_JERK_BOUND;
    const TORQUE_BOUND: [f64; N] = R::TORQUE_BOUND;
    const TORQUE_DOT_BOUND: [f64; N] = R::TORQUE_DOT_BOUND;
}

impl<const N: usize, R> ArmPreplannedMotion<N> for RsBulletRobot<R>
where
    R: ArmParam<N>,
{
    fn move_joint(&mut self, target: &[f64; N]) -> RobotResult<()> {
        self.move_joint_async(target)
    }

    fn move_joint_async(&mut self, target: &[f64; N]) -> RobotResult<()> {
        let body_id = self.body_id;
        let joint_indices = self.joint_indices.clone();

        let state = self.state()?;
        let (path_generate, t_max) = path_generate::joint_s_curve(
            &state.joint.unwrap_or([0.; N]),
            target,
            &R::JOINT_VEL_BOUND,
            &R::JOINT_ACC_BOUND,
            &R::JOINT_JERK_BOUND,
        );

        let mut duration = Duration::from_secs(0);
        self.enqueue(move |client, dt| {
            duration += dt;
            let target = path_generate(duration);
            client.set_joint_motor_control_array(
                body_id,
                &joint_indices[1..=N],
                ControlModeArray::Position(&target),
                None,
            )?;

            Ok(duration >= t_max)
        })
        .map_err(Into::into)
    }

    fn move_cartesian(&mut self, target: &Pose) -> RobotResult<()> {
        self.move_cartesian_async(target)
    }

    fn move_cartesian_async(&mut self, target: &Pose) -> RobotResult<()> {
        if self.end_effector_link < 0 {
            return Err(RobotException::UnprocessableInstructionError(
                "robot has no movable joints".to_string(),
            ));
        }

        let body_id = self.body_id;
        let joint_indices = self.joint_indices.clone();
        let end_effector_link = self.end_effector_link;

        let state = self.state()?;
        let (path_generate, t_max) = path_generate::cartesian_quat_simple_4th_curve(
            state.pose_o_to_ee.unwrap_or_default().quat(),
            target.quat(),
            R::ROTATION_VEL_BOUND,
            R::ROTATION_ACC_BOUND,
        );

        let mut duration = Duration::from_secs(0);
        self.enqueue(move |client, dt| {
            duration += dt;

            let target_pose = path_generate(duration);
            let (target_position, _) = isometry_to_raw_parts(&target_pose);

            let ik_options = InverseKinematicsOptions::<'_> {
                ..Default::default()
            };

            let solution = client.calculate_inverse_kinematics(
                body_id,
                end_effector_link,
                target_position,
                &ik_options,
            )?;

            if solution.len() < N {
                return Err(BulletError::CommandFailed {
                    message: "inverse kinematics returned insufficient joint values",
                    code: solution.len() as i32,
                });
            }

            client.set_joint_motor_control_array(
                body_id,
                &joint_indices[1..=N],
                ControlModeArray::Position(&solution[..N]),
                None,
            )?;

            Ok(duration >= t_max)
        })
        .map_err(Into::into)
    }
}

impl<const N: usize, R> ArmPreplannedPath<N> for RsBulletRobot<R>
where
    R: ArmParam<N>,
{
    fn move_traj(&mut self, path: Vec<robot_behavior::MotionType<N>>) -> RobotResult<()> {
        self.move_traj_async(path)
    }
    fn move_traj_async(&mut self, path: Vec<robot_behavior::MotionType<N>>) -> RobotResult<()> {
        let body_id = self.body_id;
        let joint_indices = self.joint_indices.clone();
        let mut path = path.into_iter();

        self.enqueue(move |client, _| match path.next() {
            Some(MotionType::Joint(joint)) => {
                client.set_joint_motor_control_array(
                    body_id,
                    &joint_indices[1..=N],
                    ControlModeArray::Position(&joint),
                    None,
                )?;
                Ok(false)
            }

            Some(MotionType::Cartesian(pose)) => {
                if joint_indices.len() < 2 {
                    return Err(BulletError::CommandFailed {
                        message: "robot has no movable joints",
                        code: -1,
                    });
                }

                let (target_position, _) = isometry_to_raw_parts(&pose.quat());

                let ik_options = InverseKinematicsOptions::<'_> {
                    ..Default::default()
                };

                let solution = client.calculate_inverse_kinematics(
                    body_id,
                    joint_indices.len() as i32 - 1,
                    target_position,
                    &ik_options,
                )?;

                if solution.len() < N {
                    return Err(BulletError::CommandFailed {
                        message: "inverse kinematics returned insufficient joint values",
                        code: solution.len() as i32,
                    });
                }

                client.set_joint_motor_control_array(
                    body_id,
                    &joint_indices[1..=N],
                    ControlModeArray::Position(&solution[..N]),
                    None,
                )?;

                Ok(false)
            }
            None => Ok(true),
            _ => Err(BulletError::CommandFailed {
                message: "unsupported motion type in preplanned path",
                code: -1,
            }),
        })
        .map_err(Into::into)
    }
}

// impl<R, const N: usize> ArmStreamingMotion<N> for RsBulletRobot<R>
// where
//     R: ArmParam<N>,
// {
//     type Handle = ;
//     fn control_with_target(&mut self) -> Arc<Mutex<Option<robot_behavior::ControlType<N>>>> {}
//     fn move_to_target(&mut self) -> Arc<Mutex<Option<robot_behavior::MotionType<N>>>> {}
//     fn end_streaming(&mut self) -> RobotResult<()> {}
//     fn start_streaming(&mut self) -> RobotResult<Self::Handle> {}
// }

impl<R, const N: usize> ArmRealtimeControl<N> for RsBulletRobot<R>
where
    R: ArmParam<N>,
{
    fn control_with_closure<FC>(&mut self, mut closure: FC) -> RobotResult<()>
    where
        FC: FnMut(ArmState<N>, Duration) -> (robot_behavior::ControlType<N>, bool) + Send + 'static,
    {
        let body_id = self.body_id;
        let joint_indices = self.joint_indices.clone();
        let cache_clone = self.state_cache.clone();
        self.enqueue(move |client, dt| {
            let state = {
                let cache = cache_clone.lock().map_err(|_| {
                    RobotException::NetworkError("state cache poisoned".to_string())
                })?;
                cache.clone().into()
            };
            let (control, done) = closure(state, dt);

            match control {
                robot_behavior::ControlType::Torque(torque) => {
                    client.set_joint_motor_control_array(
                        body_id,
                        &joint_indices[1..=N],
                        ControlModeArray::Torque(&torque),
                        None,
                    )?;
                }
                robot_behavior::ControlType::Zero => {
                    let zero_torque = vec![0.0; N];
                    client.set_joint_motor_control_array(
                        body_id,
                        &joint_indices[1..=N],
                        ControlModeArray::Torque(&zero_torque),
                        None,
                    )?;
                }
            }

            Ok(done)
        })
        .map_err(Into::into)
    }
    fn move_with_closure<FM>(&mut self, mut closure: FM) -> RobotResult<()>
    where
        FM: FnMut(ArmState<N>, Duration) -> (robot_behavior::MotionType<N>, bool) + Send + 'static,
    {
        let body_id = self.body_id;
        let joint_indices = self.joint_indices.clone();
        let cache_clone = self.state_cache.clone();
        self.enqueue(move |client, dt| {
            let state = {
                let cache = cache_clone.lock().map_err(|_| {
                    RobotException::NetworkError("state cache poisoned".to_string())
                })?;
                cache.clone().into()
            };
            let (motion, done) = closure(state, dt);

            match motion {
                robot_behavior::MotionType::Position(target) => {
                    client.set_joint_motor_control_array(
                        body_id,
                        &joint_indices[1..=N],
                        ControlModeArray::Position(&target),
                        None,
                    )?;
                }
                robot_behavior::MotionType::JointVel(vel) => {
                    client.set_joint_motor_control_array(
                        body_id,
                        &joint_indices[1..=N],
                        ControlModeArray::Velocity(&vel),
                        None,
                    )?;
                }
                _ => {}
            }

            Ok(done)
        })
        .map_err(Into::into)
    }
}
