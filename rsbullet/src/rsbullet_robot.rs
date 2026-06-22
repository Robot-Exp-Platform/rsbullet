use std::sync::{Arc, Mutex};
use std::time::Duration;
use std::{marker::PhantomData, sync::mpsc::Sender};

use anyhow::Result;
use robot_behavior::behavior::{Arm, EndPoint, FlangeSpace, JointSpace, Joints};
use robot_behavior::utils::path_generate;
use robot_behavior::{
    ArmState, ArmTorqueControl, CartesianVelocityControl, ControlWith, JointPositionControl,
    JointVelocityControl, LoadState, MoveTo, MoveTraj, Pose, Robot, RobotDescription,
    RobotException, RobotResult, TorqueControl, behavior::*,
};
use rsbullet_core::{
    BulletError, BulletResult, ControlModeArray, JointType, LoadModelFlags, PhysicsClient,
    UrdfOptions,
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

impl<'a, R: RobotDescription> RsBulletRobotBuilder<'a, R> {
    pub fn new(rsbullet: &'a mut RsBullet) -> Self {
        RsBulletRobotBuilder {
            _marker: PhantomData,
            rsbullet,
            load_file: R::URDF.expect("robot description must provide a URDF path"),
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

    /// Explicitly set the end-effector link index.
    pub fn end_effector_link(mut self, link_index: i32) -> Self {
        self.end_effector_link = Some(link_index);
        self
    }

    /// Set the end-effector link by link name.
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

impl<const N: usize, R> Arm<N> for RsBulletRobot<R>
where
    R: Joints<N> + EndPoint,
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

    fn get_joint(&self) -> [f64; N] {
        self.state_cache
            .lock()
            .ok()
            .map(|cache| Into::<ArmState<N>>::into(cache.clone()))
            .and_then(|state| state.joint.meas.q)
            .unwrap_or([0.; N])
    }

    fn get_endpoint(&self) -> Pose {
        self.state_cache
            .lock()
            .ok()
            .map(|cache| Into::<ArmState<N>>::into(cache.clone()))
            .and_then(|state| state.flange.meas.pose)
            .unwrap_or_default()
    }

    fn with_joint_vel(self, _vel_bound: [f64; N]) -> Self {
        self
    }
    fn with_joint_acc(self, _acc_bound: [f64; N]) -> Self {
        self
    }
    fn with_joint_jerk(self, _jerk_bound: [f64; N]) -> Self {
        self
    }
    fn with_torque(self, _torque_bound: [f64; N]) -> Self {
        self
    }
    fn with_torque_dot(self, _torque_dot_bound: [f64; N]) -> Self {
        self
    }
    fn with_cartesian_vel(self, _vel_bound: f64) -> Self {
        self
    }
    fn with_cartesian_acc(self, _acc_bound: f64) -> Self {
        self
    }
    fn with_cartesian_jerk(self, _jerk_bound: f64) -> Self {
        self
    }
    fn with_rotation_vel(self, _vel_bound: f64) -> Self {
        self
    }
    fn with_rotation_acc(self, _acc_bound: f64) -> Self {
        self
    }
    fn with_rotation_jerk(self, _jerk_bound: f64) -> Self {
        self
    }
}

impl<R> Robot for RsBulletRobot<R> {
    type State = RsBulletRobotState;
    const CONTROL_PERIOD: f64 = 1.0 / 240.0;

    fn version() -> String {
        "RsBulletRobot".to_string()
    }

    fn read_state(&mut self) -> RobotResult<Self::State> {
        self.state_cache
            .lock()
            .map(|cache| cache.clone())
            .map_err(|_| RobotException::NetworkError("state cache poisoned".to_string()))
    }
}

impl<const N: usize, R> Joints<N> for RsBulletRobot<R>
where
    R: Joints<N>,
{
    const JOINT_DEFAULT: [f64; N] = R::JOINT_DEFAULT;
    const JOINT_PACKED: [f64; N] = R::JOINT_PACKED;
    const JOINT_MIN: [f64; N] = R::JOINT_MIN;
    const JOINT_MAX: [f64; N] = R::JOINT_MAX;
    const JOINT_VEL_BOUND: [f64; N] = R::JOINT_VEL_BOUND;
    const JOINT_ACC_BOUND: [f64; N] = R::JOINT_ACC_BOUND;
    const JOINT_JERK_BOUND: [f64; N] = R::JOINT_JERK_BOUND;
    const TORQUE_BOUND: [f64; N] = R::TORQUE_BOUND;
    const TORQUE_DOT_BOUND: [f64; N] = R::TORQUE_DOT_BOUND;
}

impl<R> EndPoint for RsBulletRobot<R>
where
    R: EndPoint,
{
    const CARTESIAN_VEL_BOUND: f64 = R::CARTESIAN_VEL_BOUND;
    const CARTESIAN_ACC_BOUND: f64 = R::CARTESIAN_ACC_BOUND;
    const CARTESIAN_JERK_BOUND: f64 = R::CARTESIAN_JERK_BOUND;
    const ROTATION_VEL_BOUND: f64 = R::ROTATION_VEL_BOUND;
    const ROTATION_ACC_BOUND: f64 = R::ROTATION_ACC_BOUND;
    const ROTATION_JERK_BOUND: f64 = R::ROTATION_JERK_BOUND;
}

impl<const N: usize, R> MoveTo<JointSpace<N>> for RsBulletRobot<R>
where
    R: Joints<N>,
{
    fn move_to(&mut self, target: [f64; N]) -> RobotResult<()> {
        let body_id = self.body_id;
        let joint_indices = self.joint_indices.clone();

        let state: ArmState<N> = self
            .state_cache
            .lock()
            .map(|cache| cache.clone().into())
            .map_err(|_| RobotException::NetworkError("state cache poisoned".to_string()))?;
        let (path_generate, t_max) = path_generate::joint_s_curve(
            &state.joint.meas.q.unwrap_or([0.; N]),
            &target,
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
}

impl<R> MoveTo<FlangeSpace> for RsBulletRobot<R>
where
    R: EndPoint,
{
    fn move_to(&mut self, target: Pose) -> RobotResult<()> {
        let _ = target;
        Err(RobotException::UnprocessableInstructionError(
            "RsBullet generic wrapper cannot infer joint count for cartesian IK; use JointSpace"
                .into(),
        ))
    }
}

impl<const N: usize, R> MoveTraj<JointSpace<N>> for RsBulletRobot<R>
where
    R: Joints<N>,
{
    fn move_traj(&mut self, path: Vec<[f64; N]>) -> RobotResult<()> {
        let body_id = self.body_id;
        let joint_indices = self.joint_indices.clone();
        let mut path = path.into_iter();

        self.enqueue(move |client, _| match path.next() {
            Some(joint) => {
                client.set_joint_motor_control_array(
                    body_id,
                    &joint_indices[0..N],
                    ControlModeArray::Position(&joint),
                    None,
                )?;
                Ok(false)
            }
            None => Ok(true),
        })
        .map_err(Into::into)
    }

    fn move_path<F>(&mut self, _path: F) -> RobotResult<()>
    where
        F: Fn(f64) -> Option<[f64; N]>,
    {
        Err(RobotException::UnprocessableInstructionError(
            "RsBullet does not plan continuous joint paths; use move_traj or move_waypoints".into(),
        ))
    }

    fn move_waypoints(&mut self, waypoints: Vec<[f64; N]>) -> RobotResult<()> {
        <Self as MoveTraj<JointSpace<N>>>::move_traj(self, waypoints)
    }
}

impl<R> MoveTraj<FlangeSpace> for RsBulletRobot<R>
where
    R: EndPoint,
{
    fn move_traj(&mut self, path: Vec<Pose>) -> RobotResult<()> {
        let _ = path;
        Err(RobotException::UnprocessableInstructionError(
            "RsBullet generic wrapper cannot infer joint count for cartesian IK; use JointSpace"
                .into(),
        ))
    }

    fn move_path<F>(&mut self, _path: F) -> RobotResult<()>
    where
        F: Fn(f64) -> Option<Pose>,
    {
        Err(RobotException::UnprocessableInstructionError(
            "RsBullet does not plan continuous cartesian paths; use move_traj or move_waypoints"
                .into(),
        ))
    }

    fn move_waypoints(&mut self, waypoints: Vec<Pose>) -> RobotResult<()> {
        <Self as MoveTraj<FlangeSpace>>::move_traj(self, waypoints)
    }
}

fn cached_arm_state<const N: usize>(
    cache: &Arc<Mutex<RsBulletRobotState>>,
) -> BulletResult<ArmState<N>> {
    cache
        .lock()
        .map(|cache| cache.clone().into())
        .map_err(|_| BulletError::CommandFailed { message: "state cache poisoned", code: -1 })
}

fn hold_joint_position<const N: usize>(state: &JointState<N>) -> [f64; N] {
    state
        .cmd
        .q
        .or(state.des.q)
        .or(state.meas.q)
        .unwrap_or([0.0; N])
}

fn hold_joint_velocity<const N: usize>(_state: &JointState<N>) -> [f64; N] {
    [0.0; N]
}

fn hold_joint_torque<const N: usize>(state: &JointState<N>) -> [f64; N] {
    state
        .cmd
        .tau
        .or(state.des.tau)
        .or(state.meas.tau)
        .unwrap_or([0.0; N])
}

impl<R, const N: usize> ControlWith<TorqueControl<N>> for RsBulletRobot<R>
where
    R: Joints<N>,
{
    fn hold_command(state: &JointState<N>) -> [f64; N] {
        hold_joint_torque(state)
    }

    fn control_with<F>(&mut self, mut closure: F) -> RobotResult<()>
    where
        F: FnMut(JointState<N>, Duration) -> ([f64; N], bool) + Send + 'static,
    {
        let body_id = self.body_id;
        let joint_indices = self.joint_indices.clone();
        let cache = self.state_cache.clone();
        let zero_velocity = vec![0.0; N];
        let zero_force = vec![0.0; N];
        let mut torque_mode_initialized = false;

        self.enqueue(move |client, dt| {
            let state = cached_arm_state(&cache)?;
            let (torque, done) = closure(state.joint, dt);
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
            Ok(done)
        })
        .map_err(Into::into)
    }
}

impl<R, const N: usize> ControlWith<ArmTorqueControl<N>> for RsBulletRobot<R>
where
    R: Joints<N>,
{
    fn hold_command(state: &ArmState<N>) -> [f64; N] {
        hold_joint_torque(&state.joint)
    }

    fn control_with<F>(&mut self, mut closure: F) -> RobotResult<()>
    where
        F: FnMut(ArmState<N>, Duration) -> ([f64; N], bool) + Send + 'static,
    {
        let body_id = self.body_id;
        let joint_indices = self.joint_indices.clone();
        let cache = self.state_cache.clone();
        let zero_velocity = vec![0.0; N];
        let zero_force = vec![0.0; N];
        let mut torque_mode_initialized = false;

        self.enqueue(move |client, dt| {
            let state = cached_arm_state(&cache)?;
            let (torque, done) = closure(state, dt);
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
            Ok(done)
        })
        .map_err(Into::into)
    }
}

impl<R, const N: usize> ControlWith<JointPositionControl<N>> for RsBulletRobot<R>
where
    R: Joints<N>,
{
    fn hold_command(state: &JointState<N>) -> [f64; N] {
        hold_joint_position(state)
    }

    fn control_with<F>(&mut self, mut closure: F) -> RobotResult<()>
    where
        F: FnMut(JointState<N>, Duration) -> ([f64; N], bool) + Send + 'static,
    {
        let body_id = self.body_id;
        let joint_indices = self.joint_indices.clone();
        let cache = self.state_cache.clone();
        self.enqueue(move |client, dt| {
            let state = cached_arm_state(&cache)?;
            let (target, done) = closure(state.joint, dt);
            client.set_joint_motor_control_array(
                body_id,
                &joint_indices[0..N],
                ControlModeArray::Position(&target),
                None,
            )?;
            Ok(done)
        })
        .map_err(Into::into)
    }
}

impl<R, const N: usize> ControlWith<JointVelocityControl<N>> for RsBulletRobot<R>
where
    R: Joints<N>,
{
    fn hold_command(state: &JointState<N>) -> [f64; N] {
        hold_joint_velocity(state)
    }

    fn control_with<F>(&mut self, mut closure: F) -> RobotResult<()>
    where
        F: FnMut(JointState<N>, Duration) -> ([f64; N], bool) + Send + 'static,
    {
        let body_id = self.body_id;
        let joint_indices = self.joint_indices.clone();
        let cache = self.state_cache.clone();
        self.enqueue(move |client, dt| {
            let state = cached_arm_state(&cache)?;
            let (velocity, done) = closure(state.joint, dt);
            client.set_joint_motor_control_array(
                body_id,
                &joint_indices[0..N],
                ControlModeArray::Velocity(&velocity),
                None,
            )?;
            Ok(done)
        })
        .map_err(Into::into)
    }
}

impl<R, const N: usize> ControlWith<CartesianVelocityControl<N>> for RsBulletRobot<R>
where
    R: Joints<N> + EndPoint,
{
    fn hold_command(_state: &ArmState<N>) -> [f64; 6] {
        [0.; 6]
    }

    fn control_with<F>(&mut self, _closure: F) -> RobotResult<()>
    where
        F: FnMut(ArmState<N>, Duration) -> ([f64; 6], bool) + Send + 'static,
    {
        Err(RobotException::UnprocessableInstructionError(
            "RsBullet does not implement cartesian velocity realtime control".into(),
        ))
    }
}

/*
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

            let q = state.joint.meas.q.unwrap_or([0.; N]);
            let dq = state.joint.meas.dq.unwrap_or([0.; N]);
            let target = {
                let t = target_clone.lock().unwrap();
                t.unwrap_or(q)
            };
            let stiffness = *stiffness_clone.lock().unwrap();
            let damping = *damping_clone.lock().unwrap();

            // PD control: 锜?= K * (q_d - q) - D * dq
            let mut torque = [0.0; N];
            for i in 0..N {
                torque[i] = stiffness[i] * (target[i] - q[i]) - damping[i] * dq[i];
            }

            // 閹殿厾鐓╅梽鎰畽閿涘矂浼╅崗宥勮雹閻喎褰傞弫锝冣偓?            #[allow(clippy::needless_range_loop)]
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

            let q = state.joint.meas.q.unwrap_or([0.; N]);
            let dq = state.joint.meas.dq.unwrap_or([0.; N]);
            let current_pose = state.flange.meas.pose.unwrap_or_default();

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

            // Cartesian force: F = K_t * 铻杧 - D_t * J * dq (translational)
            //                   锜縚c = K_r * 铻栬儍 - D_r * J_r * dq (rotational)
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

            // Map wrench to joint torques: 锜?= J_lin^T * F + J_ang^T * 锜縚cart
            let tau = j_lin.transpose() * force + j_ang.transpose() * torque_cart;

            let mut torque = [0.0; N];
            for i in 0..N.min(tau.len()) {
                torque[i] = tau[i];
            }

            // 閹殿厾鐓╅梽鎰畽閿涘矂浼╅崗宥嗗付閸掕泛濮忔稉宥堝喕閹存牞绻冩径褔娓块懡掳鈧?            #[allow(clippy::needless_range_loop)]
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
*/
