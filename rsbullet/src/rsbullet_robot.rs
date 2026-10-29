use std::any::Any;
use std::ops::ControlFlow;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use std::{marker::PhantomData, sync::mpsc::Sender};

use anyhow::Result;
use roplat_behavior::behavior::{Arm, EndPoint, FlangeSpace, JointSpace, Joints};
use roplat_behavior::{
    ArmState, ArmTorqueControl, ControlObservation, ControlObserver, ControlSpace, ControlStep,
    JointLimits, JointPositionControl, JointState, JointVelocityControl, LoadState, MotionLimits,
    MoveTo, MoveTraj, Pose, Robot, RobotDescription, RobotException, RobotResult, TorqueControl,
};
use roplat_trajectory::path_generate;
use roplat_world::{EntityBuilder, ModelSelectionError, PhysicsEngine, SimulationBackend};
use rsbullet_core::{
    BulletError, BulletResult, ControlModeArray, JointType, LoadModelFlags, PhysicsClient,
    UrdfOptions,
};

use crate::RsBullet;
use crate::types::{QueuedControl, RsBulletRobotState};

type RsBulletControlObservers = Arc<Mutex<Vec<ControlObserver<RsBulletRobotState>>>>;

fn control_observers() -> RsBulletControlObservers {
    Arc::new(Mutex::new(Vec::new()))
}

fn notify_control_observers(
    observers: &RsBulletControlObservers,
    state: &RsBulletRobotState,
    duration: Duration,
) {
    let mut observers = observers
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    for observer in observers.iter_mut() {
        observer(state, duration);
    }
}

/// A handle into one owned simulation world.
///
/// Fallible `Robot::read_state` and `Arm::state` return the latest successfully
/// refreshed native state and reject a handle after engine fault/reset/drop.
/// The infallible `Arm::get_joint` and `EndPoint::get_endpoint` conveniences
/// cannot report validity and return their fallback values on failure; do not
/// use them as measurement evidence. Queued motions ([`MoveTo`], [`MoveTraj`])
/// use [`Arm::motion_limits`]; control sessions clamp to the declared limits.
pub struct RsBulletRobot<R> {
    pub body_id: i32,
    pub joint_indices: Vec<i32>,
    pub joint_names: Vec<String>,
    pub(crate) command_sender: Sender<QueuedControl>,
    pub end_effector_link: i32,
    state_cache: Arc<Mutex<RsBulletRobotState>>,
    pub(crate) world_validity: Arc<AtomicBool>,
    before_observers: RsBulletControlObservers,
    after_observers: RsBulletControlObservers,
    /// Legacy metadata only; this field does not sleep or change the physics dt.
    /// Configure `RsBullet::set_step_time` and `SimRhythm` pacing explicitly.
    pub control_period: Duration,
    /// The `MotionLimits<N>` set through [`Arm::set_motion_limits`], or `None`
    /// for the declared envelope. Type-erased because the handle is not
    /// generic over the joint count.
    motion_limits: Option<Box<dyn Any + Send + Sync>>,
    _marker: PhantomData<R>,
}

impl<R> RsBulletRobot<R> {
    /// False after the owning engine faults, resets, or is dropped. Reload the
    /// robot in a valid world; an old body ID must not address a replacement body.
    pub fn is_valid(&self) -> bool {
        self.world_validity.load(Ordering::Acquire)
    }

    /// The limits queued motions plan with: the motion limits at their scale,
    /// or the declared envelope.
    pub(crate) fn planning_limits<const N: usize>(&self) -> JointLimits<N>
    where
        R: Joints<N>,
    {
        self.motion_limits
            .as_deref()
            .and_then(|limits| limits.downcast_ref::<MotionLimits<N>>())
            .map_or_else(R::joint_limits, MotionLimits::scaled_joint)
    }

    /// The flange pose in the latest refreshed state.
    fn flange_pose(&self) -> RobotResult<Pose> {
        self.ensure_valid()?;
        let cache = self
            .state_cache
            .lock()
            .map_err(|_| RobotException::NetworkError("state cache poisoned".to_string()))?;
        let link = cache.link_state.as_ref().ok_or_else(|| {
            RobotException::UnprocessableInstructionError(
                "this robot has no end-effector link".into(),
            )
        })?;
        Ok(link.world_link_frame.into())
    }

    fn ensure_valid(&self) -> RobotResult<()> {
        if self.is_valid() {
            Ok(())
        } else {
            Err(RobotException::CommandException(
                "simulation robot handle is invalid after engine fault, reset, or shutdown".into(),
            ))
        }
    }

    /// Register a callback executed before native integration. Return `true` to
    /// remove it after this invocation, or `false` to run it at the next step.
    /// Use the client for queries and commands, never to integrate/reset the world;
    /// those transitions and their accounting belong to the owning engine.
    pub fn enqueue<CF>(&self, control: CF) -> anyhow::Result<()>
    where
        CF: FnMut(&mut PhysicsClient, Duration) -> BulletResult<bool> + Send + 'static,
    {
        self.ensure_valid()?;
        self.command_sender
            .send(Box::new(control))
            .map_err(|_| anyhow::anyhow!("Failed to send control command: channel closed"))?;
        Ok(())
    }

    /// Register a queued control task for this simulated robot.
    ///
    /// This queued form is rsbullet-specific and is not
    /// [`roplat_behavior::ControlWith`]: it only submits the control closure to
    /// the simulation queue; it does not step the physics engine or wait for
    /// completion. The task is consumed by [`RsBullet::step`]. For a blocking
    /// session that steps the world, use [`crate::RsBulletDevice`].
    /// The `before` / `after` hooks bracket the controller and observe the same
    /// pre-integration sample. Use [`RsBullet::add_step_observer`] for post-step
    /// object/contact evidence, or `read_state` after a successful engine step.
    pub fn control_with<S, F>(&self, mut controller: F) -> RobotResult<()>
    where
        S: RsBulletControlSpace<R> + Send + 'static,
        S::Obs: Send + 'static,
        S::Command: Send + 'static,
        F: FnMut(S::Obs, Duration) -> (S::Command, bool) + Send + 'static,
    {
        let body_id = self.body_id;
        let joint_indices = self.joint_indices.clone();
        let end_effector_link = self.end_effector_link;
        let state_cache = self.state_cache.clone();
        let before_observers = self.before_observers.clone();
        let after_observers = self.after_observers.clone();
        let mut initialized = false;

        self.enqueue(move |client, dt| {
            if !initialized {
                S::initialize(client, body_id, &joint_indices)?;
                initialized = true;
            }

            let obs = S::observe(
                client,
                body_id,
                &joint_indices,
                end_effector_link,
                &state_cache,
            )?;
            let full_state = state_cache
                .lock()
                .map_err(|_| bullet_error("state cache poisoned"))?
                .clone();
            notify_control_observers(&before_observers, &full_state, dt);
            let (command, done) = controller(obs, dt);
            notify_control_observers(&after_observers, &full_state, dt);
            S::apply(client, body_id, &joint_indices, command)?;
            Ok(done)
        })
        .map_err(|error| RobotException::CommandException(error.to_string()))
    }

    pub(crate) fn belongs_to(&self, engine: &RsBullet) -> bool {
        Arc::ptr_eq(&self.world_validity, &engine.world_validity())
    }
}

/// One control session of a robot over its own world, shared by the blocking
/// and asynchronous sessions of [`crate::RsBulletDevice`].
///
/// A cycle is [`Session::observe`], the controller, then [`Session::conclude`]:
/// the command is applied and the world integrates once. `Break` applies
/// nothing; `done` applies its command and integrates once more. However the
/// session ends, [`Session::end`] leaves the joints holding their measured
/// position, as a real arm does after a control session.
pub(crate) struct Session<'a, R, S> {
    robot: &'a RsBulletRobot<R>,
    engine: &'a mut RsBullet,
    joint_indices: Vec<i32>,
    full_state: RsBulletRobotState,
    _channel: PhantomData<fn() -> S>,
}

impl<'a, R, S: RsBulletControlSpace<R>> Session<'a, R, S> {
    pub(crate) fn begin(
        robot: &'a RsBulletRobot<R>,
        engine: &'a mut RsBullet,
    ) -> RobotResult<Self> {
        robot.ensure_valid()?;
        if !robot.belongs_to(engine) {
            return Err(RobotException::CommandException(
                "simulation robot handle belongs to another world".into(),
            ));
        }
        let joint_indices = robot.joint_indices.clone();
        S::initialize(engine.client_mut(), robot.body_id, &joint_indices)?;
        Ok(Self {
            robot,
            engine,
            joint_indices,
            full_state: RsBulletRobotState::default(),
            _channel: PhantomData,
        })
    }

    /// The physics timestep, passed to the controller every cycle.
    pub(crate) fn period(&self) -> Duration {
        self.engine.step_time()
    }

    /// Observe the channel's native state and notify the `before` observers.
    pub(crate) fn observe(&mut self) -> RobotResult<S::Obs> {
        let obs = S::observe(
            self.engine.client_mut(),
            self.robot.body_id,
            &self.joint_indices,
            self.robot.end_effector_link,
            &self.robot.state_cache,
        )?;
        self.full_state = self
            .robot
            .state_cache
            .lock()
            .map_err(|_| bullet_error("state cache poisoned"))?
            .clone();
        notify_control_observers(
            &self.robot.before_observers,
            &self.full_state,
            self.period(),
        );
        Ok(obs)
    }

    /// Notify the `after` observers, then apply a computed command and
    /// integrate once. Returns whether the session continues.
    pub(crate) fn conclude(&mut self, step: ControlStep<S::Command>) -> RobotResult<bool> {
        notify_control_observers(&self.robot.after_observers, &self.full_state, self.period());
        let ControlFlow::Continue((command, done)) = step else {
            return Ok(false);
        };
        S::apply(
            self.engine.client_mut(),
            self.robot.body_id,
            &self.joint_indices,
            command,
        )?;
        self.engine.step()?;
        Ok(!done)
    }

    /// Hold the joints, unless the world faulted, and report the outcome.
    pub(crate) fn end(self, outcome: RobotResult<()>) -> RobotResult<()> {
        if self.engine.is_faulted() || !self.robot.is_valid() {
            return outcome;
        }
        let held = S::release(
            self.engine.client_mut(),
            self.robot.body_id,
            &self.joint_indices,
        );
        outcome.and(held.map_err(Into::into))
    }
}

pub struct RsBulletRobotBuilder<'a, R> {
    pub(crate) _marker: PhantomData<R>,
    pub(crate) rsbullet: &'a mut RsBullet,
    pub(crate) load_file: std::result::Result<&'static str, ModelSelectionError>,
    pub(crate) base: Option<nalgebra::Isometry3<f64>>,
    pub(crate) base_fixed: bool,
    pub(crate) use_maximal_coordinates: Option<bool>,
    pub(crate) scaling: Option<f64>,
    pub(crate) flags: Option<LoadModelFlags>,
    pub(crate) end_effector_link: Option<i32>,
    pub(crate) end_effector_link_name: Option<String>,
    controlled_joints: Option<ControlledJoints>,
}

enum ControlledJoints {
    Indices(Vec<i32>),
    Names(Vec<String>),
}

impl<'a, R: RobotDescription> RsBulletRobotBuilder<'a, R> {
    pub fn new(rsbullet: &'a mut RsBullet) -> Self {
        RsBulletRobotBuilder {
            _marker: PhantomData,
            rsbullet,
            load_file: RsBullet::select_robot_model::<R>().map(|selected| selected.model.path),
            base: None,
            base_fixed: false,
            scaling: None,
            flags: None,
            use_maximal_coordinates: None,
            end_effector_link: None,
            end_effector_link_name: None,
            controlled_joints: None,
        }
    }
}

impl<R> RsBulletRobotBuilder<'_, R> {
    /// Select and order the movable joints controlled by this handle.
    /// Useful for an arm model that also contains separately controlled fingers.
    /// Duplicate, fixed, or absent joints are rejected when loading.
    pub fn joint_indices(mut self, indices: impl Into<Vec<i32>>) -> Self {
        self.controlled_joints = Some(ControlledJoints::Indices(indices.into()));
        self
    }

    /// Select controlled joints by their URDF joint names, in command-array order.
    pub fn joint_names<I, S>(mut self, names: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.controlled_joints = Some(ControlledJoints::Names(
            names.into_iter().map(Into::into).collect(),
        ));
        self
    }

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
        anyhow::ensure!(
            !self.rsbullet.is_faulted(),
            "reset the faulted simulation before loading robots"
        );
        let load_file = self.load_file?;
        let body_id = self.rsbullet.client_mut().load_urdf(
            load_file,
            Some(UrdfOptions {
                base: self.base,
                use_fixed_base: self.base_fixed,
                global_scaling: self.scaling,
                flags: self.flags,
                use_maximal_coordinates: self.use_maximal_coordinates,
            }),
        )?;

        let setup = (|| -> Result<_> {
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

            if let Some(selection) = self.controlled_joints {
                let selected = match selection {
                    ControlledJoints::Indices(indices) => indices,
                    ControlledJoints::Names(names) => names
                        .iter()
                        .map(|name| {
                            joint_names
                                .iter()
                                .position(|found| found == name)
                                .map(|index| joint_indices[index])
                                .ok_or_else(|| {
                                    anyhow::anyhow!(
                                        "controlled joint {name:?} is absent or not movable"
                                    )
                                })
                        })
                        .collect::<Result<Vec<_>>>()?,
                };
                let mut unique = std::collections::HashSet::new();
                let mut selected_names = Vec::with_capacity(selected.len());
                for index in &selected {
                    anyhow::ensure!(
                        unique.insert(*index),
                        "duplicate controlled joint index {index}"
                    );
                    let position = joint_indices
                        .iter()
                        .position(|found| found == index)
                        .ok_or_else(|| {
                            anyhow::anyhow!(
                                "controlled joint index {index} is absent or not movable"
                            )
                        })?;
                    selected_names.push(joint_names[position].clone());
                }
                joint_indices = selected;
                joint_names = selected_names;
            }

            let end_effector_link = if let Some(idx) = self.end_effector_link {
                anyhow::ensure!(
                    idx >= -1 && idx < joint_count,
                    "end-effector link index {idx} is outside the loaded model"
                );
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
                selected.ok_or_else(|| {
                    anyhow::anyhow!("end-effector link {name:?} is absent from the loaded model")
                })?
            } else if joint_count == 0 {
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
            Ok((
                joint_indices,
                joint_names,
                end_effector_link,
                joint_states,
                link_state,
            ))
        })();
        let (joint_indices, joint_names, end_effector_link, joint_states, link_state) = match setup
        {
            Ok(state) => state,
            Err(error) => {
                // Failed configuration must not leave an untracked loaded body behind.
                if let Err(cleanup) = self.rsbullet.client_mut().remove_body(body_id) {
                    return Err(
                        error.context(format!("also failed to remove incomplete robot: {cleanup}"))
                    );
                }
                return Err(error);
            }
        };
        let state_cache = Arc::new(Mutex::new(RsBulletRobotState {
            joint_states,
            link_state,
        }));

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
            world_validity: self.rsbullet.world_validity(),
            before_observers: control_observers(),
            after_observers: control_observers(),
            control_period: Duration::from_secs_f64(1.0 / 240.0),
            motion_limits: None,
            _marker: PhantomData,
        };

        self.rsbullet
            .register_state_refresh(Box::new(move |client| {
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
                Ok(())
            }));

        Ok(robot)
    }
}

impl<const N: usize, R> Arm<N> for RsBulletRobot<R>
where
    R: Joints<N> + EndPoint,
{
    fn state(&mut self) -> RobotResult<ArmState<N>> {
        self.ensure_valid()?;
        checked_joints::<N>(&self.joint_indices)?;
        let cache = self
            .state_cache
            .lock()
            .map_err(|_| RobotException::NetworkError("state cache poisoned".to_string()))?;
        Ok(cache.clone().into())
    }

    fn set_load(&mut self, _load: LoadState) -> RobotResult<()> {
        Err(RobotException::UnprocessableInstructionError(
            "RsBullet does not apply payload inertia through Arm::set_load; configure the physical model explicitly".into(),
        ))
    }

    fn get_joint(&self) -> [f64; N] {
        // This legacy infallible convenience API has no validity channel.
        // Use `state` / `read_state` for measurements and error handling.
        if !self.is_valid() {
            return [0.; N];
        }
        self.state_cache
            .lock()
            .ok()
            .map(|cache| Into::<ArmState<N>>::into(cache.clone()))
            .and_then(|state| state.joint.meas.q())
            .unwrap_or([0.; N])
    }

    fn motion_limits(&self) -> MotionLimits<N> {
        self.motion_limits
            .as_deref()
            .and_then(|limits| limits.downcast_ref::<MotionLimits<N>>())
            .copied()
            .unwrap_or_else(MotionLimits::envelope::<R>)
    }

    fn set_motion_limits(&mut self, limits: MotionLimits<N>) -> RobotResult<()> {
        limits.within(&MotionLimits::envelope::<R>())?;
        self.motion_limits = Some(Box::new(limits));
        Ok(())
    }
}

impl<R> Robot for RsBulletRobot<R> {
    type State = RsBulletRobotState;
    const CONTROL_PERIOD: Duration = Duration::from_nanos(1_000_000_000 / 240);

    fn version() -> String {
        "RsBulletRobot".to_string()
    }

    fn read_state(&mut self) -> RobotResult<Self::State> {
        self.ensure_valid()?;
        self.state_cache
            .lock()
            .map(|cache| cache.clone())
            .map_err(|_| RobotException::NetworkError("state cache poisoned".to_string()))
    }
}

impl<R> ControlObservation for RsBulletRobot<R> {
    fn before<H>(&mut self, observer: H) -> &mut Self
    where
        H: FnMut(&Self::State, Duration) + Send + 'static,
    {
        self.before_observers
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .push(Box::new(observer));
        self
    }

    fn after<H>(&mut self, observer: H) -> &mut Self
    where
        H: FnMut(&Self::State, Duration) + Send + 'static,
    {
        self.after_observers
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .push(Box::new(observer));
        self
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

    fn get_endpoint(&self) -> Pose {
        self.flange_pose().unwrap_or_default()
    }

    fn read_endpoint(&mut self) -> RobotResult<Pose> {
        self.flange_pose()
    }
}

impl<const N: usize, R> MoveTo<JointSpace<N>> for RsBulletRobot<R>
where
    R: Joints<N>,
{
    fn move_to(&mut self, target: [f64; N]) -> RobotResult<()> {
        self.ensure_valid()?;
        checked_joints::<N>(&self.joint_indices)?;
        let planned = self.planning_limits::<N>();
        let target = clamp_position(target, planned.q_min, planned.q_max)?;
        let body_id = self.body_id;
        let joint_indices = self.joint_indices.clone();

        let state: ArmState<N> = self
            .state_cache
            .lock()
            .map(|cache| cache.clone().into())
            .map_err(|_| RobotException::NetworkError("state cache poisoned".to_string()))?;
        let (path_generate, t_max) = path_generate::joint_s_curve(
            &state.joint.meas.q().unwrap_or([0.; N]),
            &target,
            &planned.dq_max,
            &planned.ddq_max,
            &planned.dddq_max,
        );

        let mut duration = Duration::from_secs(0);
        self.enqueue(move |client, dt| {
            duration += dt;
            let target = path_generate(duration);
            let limits = joint_effort_limits(client, body_id, &joint_indices, planned.tau_max)?;
            client.set_joint_motor_control_array(
                body_id,
                checked_joints::<N>(&joint_indices)?,
                ControlModeArray::Position(&target),
                Some(&limits),
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
        self.ensure_valid()?;
        checked_joints::<N>(&self.joint_indices)?;
        let body_id = self.body_id;
        let joint_indices = self.joint_indices.clone();
        let planned = self.planning_limits::<N>();
        let mut path = path.into_iter();

        self.enqueue(move |client, _| match path.next() {
            Some(joint) => {
                let joint = clamp_position(joint, planned.q_min, planned.q_max)?;
                let limits = joint_effort_limits(client, body_id, &joint_indices, planned.tau_max)?;
                client.set_joint_motor_control_array(
                    body_id,
                    checked_joints::<N>(&joint_indices)?,
                    ControlModeArray::Position(&joint),
                    Some(&limits),
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

/// A rsbullet-specific mapping from a roplat_behavior control space to Bullet
/// observations and commands.
///
/// This trait is local to rsbullet. It serves both the queued
/// [`RsBulletRobot::control_with`] and the blocking `ControlWith` sessions of
/// [`crate::RsBulletDevice`].
pub trait RsBulletControlSpace<R>: ControlSpace<RsBulletRobot<R>> {
    fn initialize(
        _client: &mut PhysicsClient,
        _body_id: i32,
        _joint_indices: &[i32],
    ) -> BulletResult<()> {
        Ok(())
    }

    fn observe(
        client: &mut PhysicsClient,
        body_id: i32,
        joint_indices: &[i32],
        end_effector_link: i32,
        state_cache: &Arc<Mutex<RsBulletRobotState>>,
    ) -> BulletResult<Self::Obs>;

    fn apply(
        client: &mut PhysicsClient,
        body_id: i32,
        joint_indices: &[i32],
        command: Self::Command,
    ) -> BulletResult<()>;

    /// Leave the joints in a resting mode when a blocking session ends. The
    /// built-in channels hold the measured joint position.
    fn release(
        _client: &mut PhysicsClient,
        _body_id: i32,
        _joint_indices: &[i32],
    ) -> BulletResult<()> {
        Ok(())
    }
}

fn bullet_error(message: &'static str) -> BulletError {
    BulletError::CommandFailed { message, code: -1 }
}

fn checked_joints<const N: usize>(joint_indices: &[i32]) -> BulletResult<&[i32]> {
    if joint_indices.len() < N {
        return Err(bullet_error(
            "robot has fewer controllable joints than requested",
        ));
    }
    Ok(&joint_indices[..N])
}

/// Intersect the effort bounds `tau_max` with finite positive URDF effort
/// limits. An unbounded (infinite) entry needs a URDF limit, since Bullet's
/// motors take a finite force.
fn joint_effort_limits<const N: usize>(
    client: &mut PhysicsClient,
    body_id: i32,
    joint_indices: &[i32],
    tau_max: [f64; N],
) -> BulletResult<[f64; N]> {
    let joints = checked_joints::<N>(joint_indices)?;
    let mut limits = tau_max;
    for (index, &joint) in joints.iter().enumerate() {
        let declared = limits[index];
        if declared.is_nan() || declared < 0.0 {
            return Err(bullet_error(
                "declared joint effort limits must be nonnegative",
            ));
        }
        let native = client.get_joint_info(body_id, joint)?.max_force;
        limits[index] = if native.is_finite() && native > 0.0 {
            declared.min(native)
        } else if declared.is_finite() {
            declared
        } else {
            return Err(bullet_error(
                "joint needs a finite URDF effort limit or an explicit TORQUE_BOUND",
            ));
        };
    }
    Ok(limits)
}

fn read_robot_state(
    client: &mut PhysicsClient,
    body_id: i32,
    joint_indices: &[i32],
    end_effector_link: i32,
    state_cache: &Arc<Mutex<RsBulletRobotState>>,
) -> BulletResult<RsBulletRobotState> {
    let joint_states = client.get_joint_states(body_id, joint_indices)?;
    let link_state = if end_effector_link >= 0 {
        Some(client.get_link_state(body_id, end_effector_link, true, true)?)
    } else {
        None
    };
    let state = RsBulletRobotState {
        joint_states,
        link_state,
    };

    let mut cache = state_cache
        .lock()
        .map_err(|_| bullet_error("state cache poisoned"))?;
    *cache = state.clone();

    Ok(state)
}

fn read_arm_state<const N: usize>(
    client: &mut PhysicsClient,
    body_id: i32,
    joint_indices: &[i32],
    end_effector_link: i32,
    cache: &Arc<Mutex<RsBulletRobotState>>,
) -> BulletResult<ArmState<N>> {
    checked_joints::<N>(joint_indices)?;
    Ok(read_robot_state(client, body_id, joint_indices, end_effector_link, cache)?.into())
}

/// An infinite limit leaves its value as it is.
fn clamp_by<const N: usize>(mut values: [f64; N], limit: [f64; N]) -> BulletResult<[f64; N]> {
    for i in 0..N {
        if !values[i].is_finite() || limit[i].is_nan() || limit[i] < 0.0 {
            return Err(bullet_error(
                "joint commands must be finite and limits nonnegative",
            ));
        }
        values[i] = values[i].clamp(-limit[i], limit[i]);
    }
    Ok(values)
}

/// An infinite bound leaves its side unclamped.
pub(crate) fn clamp_position<const N: usize>(
    mut values: [f64; N],
    lower: [f64; N],
    upper: [f64; N],
) -> BulletResult<[f64; N]> {
    for i in 0..N {
        if !values[i].is_finite() || lower[i].is_nan() || upper[i].is_nan() || lower[i] > upper[i] {
            return Err(bullet_error(
                "joint positions must be finite and limits ordered",
            ));
        }
        values[i] = values[i].clamp(lower[i], upper[i]);
    }
    Ok(values)
}

fn disable_default_motor<const N: usize>(
    client: &mut PhysicsClient,
    body_id: i32,
    joint_indices: &[i32],
) -> BulletResult<()> {
    let joints = checked_joints::<N>(joint_indices)?;
    let zero_velocity = [0.0; N];
    let zero_force = [0.0; N];
    client.set_joint_motor_control_array(
        body_id,
        joints,
        ControlModeArray::Velocity(&zero_velocity),
        Some(&zero_force),
    )
}

/// Hold the measured position with the channel's effort limits.
fn hold_measured_position<R: Joints<N>, const N: usize>(
    client: &mut PhysicsClient,
    body_id: i32,
    joint_indices: &[i32],
) -> BulletResult<()> {
    let joints = checked_joints::<N>(joint_indices)?;
    let states = client.get_joint_states(body_id, joints)?;
    if states.len() < N {
        return Err(bullet_error(
            "incomplete joint state while holding position",
        ));
    }
    let measured = std::array::from_fn(|i| states[i].position);
    let declared = R::joint_limits();
    let target = clamp_position(measured, declared.q_min, declared.q_max)?;
    let limits = joint_effort_limits(client, body_id, joint_indices, declared.tau_max)?;
    client.set_joint_motor_control_array(
        body_id,
        joints,
        ControlModeArray::Position(&target),
        Some(&limits),
    )
}

impl<R, const N: usize> RsBulletControlSpace<R> for JointPositionControl<N>
where
    R: Joints<N>,
{
    fn observe(
        client: &mut PhysicsClient,
        body_id: i32,
        joint_indices: &[i32],
        end_effector_link: i32,
        state_cache: &Arc<Mutex<RsBulletRobotState>>,
    ) -> BulletResult<JointState<N>> {
        Ok(read_arm_state::<N>(
            client,
            body_id,
            joint_indices,
            end_effector_link,
            state_cache,
        )?
        .joint)
    }

    fn apply(
        client: &mut PhysicsClient,
        body_id: i32,
        joint_indices: &[i32],
        command: [f64; N],
    ) -> BulletResult<()> {
        let joints = checked_joints::<N>(joint_indices)?;
        let declared = R::joint_limits();
        let command = clamp_position(command, declared.q_min, declared.q_max)?;
        let limits = joint_effort_limits(client, body_id, joint_indices, declared.tau_max)?;
        client.set_joint_motor_control_array(
            body_id,
            joints,
            ControlModeArray::Position(&command),
            Some(&limits),
        )
    }

    fn release(
        client: &mut PhysicsClient,
        body_id: i32,
        joint_indices: &[i32],
    ) -> BulletResult<()> {
        hold_measured_position::<R, N>(client, body_id, joint_indices)
    }
}

impl<R, const N: usize> RsBulletControlSpace<R> for JointVelocityControl<N>
where
    R: Joints<N>,
{
    fn observe(
        client: &mut PhysicsClient,
        body_id: i32,
        joint_indices: &[i32],
        end_effector_link: i32,
        state_cache: &Arc<Mutex<RsBulletRobotState>>,
    ) -> BulletResult<JointState<N>> {
        Ok(read_arm_state::<N>(
            client,
            body_id,
            joint_indices,
            end_effector_link,
            state_cache,
        )?
        .joint)
    }

    fn apply(
        client: &mut PhysicsClient,
        body_id: i32,
        joint_indices: &[i32],
        command: [f64; N],
    ) -> BulletResult<()> {
        let joints = checked_joints::<N>(joint_indices)?;
        let declared = R::joint_limits();
        let command = clamp_by(command, declared.dq_max)?;
        let limits = joint_effort_limits(client, body_id, joint_indices, declared.tau_max)?;
        client.set_joint_motor_control_array(
            body_id,
            joints,
            ControlModeArray::Velocity(&command),
            Some(&limits),
        )
    }

    fn release(
        client: &mut PhysicsClient,
        body_id: i32,
        joint_indices: &[i32],
    ) -> BulletResult<()> {
        hold_measured_position::<R, N>(client, body_id, joint_indices)
    }
}

impl<R, const N: usize> RsBulletControlSpace<R> for TorqueControl<N>
where
    R: Joints<N>,
{
    fn initialize(
        client: &mut PhysicsClient,
        body_id: i32,
        joint_indices: &[i32],
    ) -> BulletResult<()> {
        disable_default_motor::<N>(client, body_id, joint_indices)
    }

    fn observe(
        client: &mut PhysicsClient,
        body_id: i32,
        joint_indices: &[i32],
        end_effector_link: i32,
        state_cache: &Arc<Mutex<RsBulletRobotState>>,
    ) -> BulletResult<JointState<N>> {
        Ok(read_arm_state::<N>(
            client,
            body_id,
            joint_indices,
            end_effector_link,
            state_cache,
        )?
        .joint)
    }

    fn apply(
        client: &mut PhysicsClient,
        body_id: i32,
        joint_indices: &[i32],
        command: [f64; N],
    ) -> BulletResult<()> {
        let joints = checked_joints::<N>(joint_indices)?;
        let limits =
            joint_effort_limits(client, body_id, joint_indices, R::joint_limits().tau_max)?;
        let command = clamp_by(command, limits)?;
        client.set_joint_motor_control_array(
            body_id,
            joints,
            ControlModeArray::Torque(&command),
            None,
        )
    }

    fn release(
        client: &mut PhysicsClient,
        body_id: i32,
        joint_indices: &[i32],
    ) -> BulletResult<()> {
        hold_measured_position::<R, N>(client, body_id, joint_indices)
    }
}

impl<R, const N: usize> RsBulletControlSpace<R> for ArmTorqueControl<N>
where
    R: Joints<N>,
{
    fn initialize(
        client: &mut PhysicsClient,
        body_id: i32,
        joint_indices: &[i32],
    ) -> BulletResult<()> {
        disable_default_motor::<N>(client, body_id, joint_indices)
    }

    fn observe(
        client: &mut PhysicsClient,
        body_id: i32,
        joint_indices: &[i32],
        end_effector_link: i32,
        state_cache: &Arc<Mutex<RsBulletRobotState>>,
    ) -> BulletResult<ArmState<N>> {
        read_arm_state::<N>(
            client,
            body_id,
            joint_indices,
            end_effector_link,
            state_cache,
        )
    }

    fn apply(
        client: &mut PhysicsClient,
        body_id: i32,
        joint_indices: &[i32],
        command: [f64; N],
    ) -> BulletResult<()> {
        let joints = checked_joints::<N>(joint_indices)?;
        let limits =
            joint_effort_limits(client, body_id, joint_indices, R::joint_limits().tau_max)?;
        let command = clamp_by(command, limits)?;
        client.set_joint_motor_control_array(
            body_id,
            joints,
            ControlModeArray::Torque(&command),
            None,
        )
    }

    fn release(
        client: &mut PhysicsClient,
        body_id: i32,
        joint_indices: &[i32],
    ) -> BulletResult<()> {
        hold_measured_position::<R, N>(client, body_id, joint_indices)
    }
}
