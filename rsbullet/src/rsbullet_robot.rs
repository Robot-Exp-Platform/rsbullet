use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use std::{marker::PhantomData, sync::mpsc::Sender};

use anyhow::Result;
use futures::future::BoxFuture;
use robot_behavior::behavior::{Arm, ArmParam};
use robot_behavior::utils::path_generate;
use robot_behavior::{
    ArmImpedance, ArmPreplannedPath, ArmState, CartesianImpedanceHandle, Coord,
    JointImpedanceHandle, JointStateSync, LoadState, MotionType, Pose, RobotException, RobotFile,
    RobotResult, behavior::*,
};
use rsbullet_core::{
    BulletError, BulletResult, ControlModeArray, InverseKinematicsOptions, JointType,
    LoadModelFlags, PhysicsClient, UrdfOptions,
};

use crate::RsBullet;
use crate::types::{QueuedControl, RsBulletRobotState};

pub struct RsBulletRobot<R> {
    pub body_id: i32,
    pub joint_indices: Vec<i32>,
    pub joint_names: Vec<String>,
    pub(crate) command_sender: Sender<QueuedControl>,
    pub end_effector_link: i32,
    state_cache: Arc<Mutex<RsBulletRobotState>>,
    /// Control period for rate-limiting simulation. If the simulation step completes
    /// faster than this duration, the step callback will sleep for the remaining time.
    /// Typically set to match the desired control frequency (e.g., 1ms for 1kHz).
    pub control_period: Duration,
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
    pub(crate) end_effector_link: Option<i32>,
    pub(crate) end_effector_link_name: Option<String>,
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
            end_effector_link: None,
            end_effector_link_name: None,
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

    /// 显式指定末端 link 索引（Bullet joint index 语义）
    pub fn end_effector_link(mut self, link_index: i32) -> Self {
        self.end_effector_link = Some(link_index);
        self
    }

    /// 按 link 名称指定末端（优先于默认推断）
    pub fn end_effector_link_name(mut self, link_name: impl Into<String>) -> Self {
        self.end_effector_link_name = Some(link_name.into());
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
        let mut joint_indices = Vec::new();
        let mut joint_names = Vec::new();
        for i in 0..joint_count {
            let info = self.rsbullet.client_mut().get_joint_info(body_id, i)?;
            if matches!(info.joint_type, JointType::Revolute | JointType::Prismatic) {
                joint_indices.push(i);
                joint_names.push(info.joint_name.clone());
            }
        }

        let end_effector_link = if let Some(idx) = self.end_effector_link {
            idx
        } else if let Some(name) = &self.end_effector_link_name {
            let mut selected = None;
            for i in 0..joint_count {
                let info = self.rsbullet.client_mut().get_joint_info(body_id, i)?;
                if info.link_name == *name {
                    selected = Some(i);
                    break;
                }
            }
            selected.unwrap_or_else(|| {
                if joint_count == 0 {
                    -1
                } else {
                    joint_count - 1
                }
            })
        } else {
            if joint_count == 0 {
                -1
            } else {
                joint_count - 1
            }
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
            joint_names,
            command_sender: sender,
            end_effector_link,
            state_cache,
            control_period: Duration::from_secs_f64(1.0 / 240.0),
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

impl<R: JointStateSync> AttachFrom<R> for RsBulletRobot<R> {
    fn attach_from(self, from: &mut R) -> Result<()> {
        let handle = from.joint_state_handle();
        let body_id = self.body_id;
        let joint_names = self.joint_names.clone();
        let joint_indices = self.joint_indices.clone();

        self.enqueue(move |client, _| {
            let map = handle.lock().map_err(|_| BulletError::CommandFailed {
                message: "joint state handle poisoned",
                code: -1,
            })?;

            for (i, name) in joint_names.iter().enumerate() {
                if let Some(entry) = map.get(name) {
                    client.reset_joint_state(
                        body_id,
                        joint_indices[i],
                        entry.position,
                        entry.velocity,
                    )?;
                }
            }

            Ok(false) // keep running every step
        })?;

        Ok(())
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

    fn set_scale(&mut self, _scale: f64) -> RobotResult<()> {
        Ok(())
    }

    fn with_scale(&mut self, _scale: f64) -> &mut Self {
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
    const CONTROL_PERIOD: f64 = R::CONTROL_PERIOD;
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
            &state.measured.joint.unwrap_or([0.; N]),
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
                &joint_indices[0..N],
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
            state.measured.pose_o_to_ee.unwrap_or_default().quat(),
            target.quat(),
            R::ROTATION_VEL_BOUND,
            R::ROTATION_ACC_BOUND,
        );

        let mut duration = Duration::from_secs(0);
        self.enqueue(move |client, dt| {
            duration += dt;

            let target_pose = path_generate(duration);

            let current_q: Vec<f64> = client
                .get_joint_states(body_id, &joint_indices)?
                .iter()
                .map(|s| s.position)
                .collect();
            let dof = client.compute_dof_count(body_id) as usize;
            let mut lower_full: Vec<f64> = R::JOINT_MIN.iter().copied().collect();
            let mut upper_full: Vec<f64> = R::JOINT_MAX.iter().copied().collect();
            let mut rest_full: Vec<f64> = R::JOINT_DEFAULT.iter().copied().collect();
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
            let mut seed = current_q;
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
                body_id,
                end_effector_link,
                target_pose,
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
                &joint_indices[0..N],
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
                    &joint_indices[0..N],
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

                let current_q: Vec<f64> = client
                    .get_joint_states(body_id, &joint_indices)?
                    .iter()
                    .map(|s| s.position)
                    .collect();
                let dof = client.compute_dof_count(body_id) as usize;
                let mut lower_full: Vec<f64> = R::JOINT_MIN.iter().copied().collect();
                let mut upper_full: Vec<f64> = R::JOINT_MAX.iter().copied().collect();
                let mut rest_full: Vec<f64> = R::JOINT_DEFAULT.iter().copied().collect();
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
                let mut seed = current_q;
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
                    body_id,
                    joint_indices.len() as i32 - 1,
                    pose.quat(),
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
                    &joint_indices[0..N],
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
        let zero_velocity = vec![0.0; N];
        let zero_force = vec![0.0; N];
        let mut torque_mode_initialized = false;
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
                    if !torque_mode_initialized {
                        client.set_joint_motor_control_array(
                            body_id,
                            &joint_indices[0..N],
                            ControlModeArray::Velocity(&zero_velocity),
                            Some(&zero_force),
                        )?;
                        torque_mode_initialized = true;
                    }
                    client.set_joint_motor_control_array(
                        body_id,
                        &joint_indices[0..N],
                        ControlModeArray::Torque(&torque),
                        None,
                    )?;
                }
                robot_behavior::ControlType::Zero => {
                    if !torque_mode_initialized {
                        client.set_joint_motor_control_array(
                            body_id,
                            &joint_indices[0..N],
                            ControlModeArray::Velocity(&zero_velocity),
                            Some(&zero_force),
                        )?;
                        torque_mode_initialized = true;
                    }
                    let zero_torque = vec![0.0; N];
                    client.set_joint_motor_control_array(
                        body_id,
                        &joint_indices[0..N],
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
        let end_effector_link = self.end_effector_link;
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
                robot_behavior::MotionType::Joint(target) => {
                    client.set_joint_motor_control_array(
                        body_id,
                        &joint_indices[0..N],
                        ControlModeArray::Position(&target),
                        None,
                    )?;
                }
                robot_behavior::MotionType::JointVel(vel) => {
                    client.set_joint_motor_control_array(
                        body_id,
                        &joint_indices[0..N],
                        ControlModeArray::Velocity(&vel),
                        None,
                    )?;
                }
                robot_behavior::MotionType::Cartesian(pose) => {
                    if end_effector_link < 0 {
                        return Err(BulletError::CommandFailed {
                            message: "robot has no end-effector link",
                            code: -1,
                        });
                    }

                    let current_q: Vec<f64> = client
                        .get_joint_states(body_id, &joint_indices)?
                        .iter()
                        .map(|s| s.position)
                        .collect();
                    let dof = client.compute_dof_count(body_id) as usize;
                    let mut lower_full: Vec<f64> = R::JOINT_MIN.iter().copied().collect();
                    let mut upper_full: Vec<f64> = R::JOINT_MAX.iter().copied().collect();
                    let mut rest_full: Vec<f64> = R::JOINT_DEFAULT.iter().copied().collect();
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
                    let mut seed = current_q;
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
                        body_id,
                        end_effector_link,
                        pose.quat(),
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
                        &joint_indices[0..N],
                        ControlModeArray::Position(&solution[..N]),
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

impl<R, const N: usize> ArmImpedance<N> for RsBulletRobot<R>
where
    R: ArmParam<N>,
{
    fn joint_impedance_async(
        &mut self,
        stiffness: &[f64; N],
        damping: &[f64; N],
    ) -> RobotResult<JointImpedanceHandle<N>> {
        let handle = JointImpedanceHandle {
            stiffness: Arc::new(Mutex::new(*stiffness)),
            damping: Arc::new(Mutex::new(*damping)),
            target: Arc::new(Mutex::new(None)),
            is_finished: Arc::new(AtomicBool::new(false)),
        };

        let stiffness_clone = handle.stiffness.clone();
        let damping_clone = handle.damping.clone();
        let target_clone = handle.target.clone();
        let is_finished_clone = handle.is_finished.clone();

        let body_id = self.body_id;
        let joint_indices = self.joint_indices.clone();
        let cache_clone = self.state_cache.clone();
        let zero_velocity = vec![0.0; N];
        let zero_force = vec![0.0; N];
        let mut torque_mode_initialized = false;

        self.enqueue(move |client, _| {
            let state: ArmState<N> = {
                let cache = cache_clone.lock().map_err(|_| BulletError::CommandFailed {
                    message: "state cache poisoned",
                    code: -1,
                })?;
                cache.clone().into()
            };

            let q = state.measured.joint.unwrap_or([0.; N]);
            let dq = state.measured.joint_vel.unwrap_or([0.; N]);
            let target = {
                let t = target_clone.lock().unwrap();
                t.unwrap_or(q)
            };
            let stiffness = *stiffness_clone.lock().unwrap();
            let damping = *damping_clone.lock().unwrap();

            // PD control: τ = K * (q_d - q) - D * dq
            let mut torque = [0.0; N];
            for i in 0..N {
                torque[i] = stiffness[i] * (target[i] - q[i]) - damping[i] * dq[i];
            }

            // 扭矩限幅，避免仿真发散。
            #[allow(clippy::needless_range_loop)]
            for i in 0..N {
                let tau = torque[i];
                let limit = R::TORQUE_BOUND[i].abs();
                torque[i] = tau.clamp(-limit, limit);
            }

            if !torque_mode_initialized {
                client.set_joint_motor_control_array(
                    body_id,
                    &joint_indices[0..N],
                    ControlModeArray::Velocity(&zero_velocity),
                    Some(&zero_force),
                )?;
                torque_mode_initialized = true;
            }

            client.set_joint_motor_control_array(
                body_id,
                &joint_indices[0..N],
                ControlModeArray::Torque(&torque),
                None,
            )?;

            Ok(is_finished_clone.load(Ordering::SeqCst))
        })
        .map_err(|e| RobotException::CommandException(format!("{e}")))?;

        Ok(handle)
    }

    fn cartesian_impedance_async(
        &mut self,
        stiffness: (f64, f64),
        damping: (f64, f64),
    ) -> RobotResult<CartesianImpedanceHandle> {
        let handle = CartesianImpedanceHandle {
            stiffness: Arc::new(Mutex::new(stiffness)),
            damping: Arc::new(Mutex::new(damping)),
            target: Arc::new(Mutex::new(None)),
            is_finished: Arc::new(AtomicBool::new(false)),
        };

        let stiffness_clone = handle.stiffness.clone();
        let damping_clone = handle.damping.clone();
        let target_clone = handle.target.clone();
        let is_finished_clone = handle.is_finished.clone();

        let body_id = self.body_id;
        let joint_indices = self.joint_indices.clone();
        let end_effector_link = self.end_effector_link;
        let cache_clone = self.state_cache.clone();
        let zero_velocity = vec![0.0; N];
        let zero_force = vec![0.0; N];
        let mut torque_mode_initialized = false;

        self.enqueue(move |client, _| {
            let state: ArmState<N> = {
                let cache = cache_clone.lock().map_err(|_| BulletError::CommandFailed {
                    message: "state cache poisoned",
                    code: -1,
                })?;
                cache.clone().into()
            };

            let q = state.measured.joint.unwrap_or([0.; N]);
            let dq = state.measured.joint_vel.unwrap_or([0.; N]);
            let current_pose = state.measured.pose_o_to_ee.unwrap_or_default();

            let target_pose = {
                let t = target_clone.lock().unwrap();
                t.unwrap_or(current_pose)
            };
            let (trans_stiffness, rot_stiffness) = *stiffness_clone.lock().unwrap();
            let (trans_damping, rot_damping) = *damping_clone.lock().unwrap();

            // Compute Jacobian via physics engine
            let zero_acc = vec![0.0; N];
            let jacobian = client.calculate_jacobian(
                body_id,
                end_effector_link,
                [0.0, 0.0, 0.0],
                &q,
                &dq,
                &zero_acc,
            )?;

            // Cartesian position/orientation error
            let current_quat = current_pose.quat();
            let target_quat = target_pose.quat();

            let pos_error = target_quat.translation.vector - current_quat.translation.vector;
            let rot_error = {
                let q_cur = current_quat.rotation;
                let q_des = target_quat.rotation;
                let q_err = q_des * q_cur.inverse();
                q_err.scaled_axis()
            };

            // Cartesian force: F = K_t * Δx - D_t * J * dq (translational)
            //                   τ_c = K_r * Δθ - D_r * J_r * dq (rotational)
            let j_lin = &jacobian.linear; // 3xN
            let j_ang = &jacobian.angular; // 3xN
            let dq_vec = nalgebra::DVector::from_column_slice(&dq);
            let lin_vel = j_lin * &dq_vec; // 3x1
            let ang_vel = j_ang * &dq_vec; // 3x1

            let mut force = nalgebra::Vector3::zeros();
            let mut torque_cart = nalgebra::Vector3::zeros();
            for i in 0..3 {
                force[i] = trans_stiffness * pos_error[i] - trans_damping * lin_vel[i];
                torque_cart[i] = rot_stiffness * rot_error[i] - rot_damping * ang_vel[i];
            }

            // Map wrench to joint torques: τ = J_lin^T * F + J_ang^T * τ_cart
            let tau = j_lin.transpose() * force + j_ang.transpose() * torque_cart;

            let mut torque = [0.0; N];
            for i in 0..N.min(tau.len()) {
                torque[i] = tau[i];
            }

            // 扭矩限幅，避免控制力不足或过大震荡。
            #[allow(clippy::needless_range_loop)]
            for i in 0..N {
                let tau = torque[i];
                let limit = R::TORQUE_BOUND[i].abs();
                torque[i] = tau.clamp(-limit, limit);
            }

            if !torque_mode_initialized {
                client.set_joint_motor_control_array(
                    body_id,
                    &joint_indices[0..N],
                    ControlModeArray::Velocity(&zero_velocity),
                    Some(&zero_force),
                )?;
                torque_mode_initialized = true;
            }

            client.set_joint_motor_control_array(
                body_id,
                &joint_indices[0..N],
                ControlModeArray::Torque(&torque),
                None,
            )?;

            Ok(is_finished_clone.load(Ordering::SeqCst))
        })
        .map_err(|e| RobotException::CommandException(format!("{e}")))?;

        Ok(handle)
    }

    fn joint_impedance_control(
        &mut self,
        stiffness: &[f64; N],
        damping: &[f64; N],
    ) -> RobotResult<(
        JointImpedanceHandle<N>,
        impl FnMut() -> BoxFuture<'static, RobotResult<()>> + Send + 'static,
    )> {
        let handle = self.joint_impedance_async(stiffness, damping)?;
        let is_finished = handle.is_finished.clone();

        let closure = Box::new(move || {
            let is_finished = is_finished.clone();
            Box::pin(async move {
                while !is_finished.load(Ordering::SeqCst) {
                    tokio::time::sleep(Duration::from_millis(10)).await;
                }
                Ok(())
            }) as BoxFuture<'static, RobotResult<()>>
        });

        Ok((handle, closure))
    }

    fn cartesian_impedance_control(
        &mut self,
        stiffness: (f64, f64),
        damping: (f64, f64),
    ) -> RobotResult<(
        CartesianImpedanceHandle,
        impl FnMut() -> BoxFuture<'static, RobotResult<()>> + Send + 'static,
    )> {
        let handle = self.cartesian_impedance_async(stiffness, damping)?;
        let is_finished = handle.is_finished.clone();

        let closure = Box::new(move || {
            let is_finished = is_finished.clone();
            Box::pin(async move {
                while !is_finished.load(Ordering::SeqCst) {
                    tokio::time::sleep(Duration::from_millis(10)).await;
                }
                Ok(())
            }) as BoxFuture<'static, RobotResult<()>>
        });

        Ok((handle, closure))
    }
}
