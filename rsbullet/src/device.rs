use std::{
    future::Future,
    ops::ControlFlow,
    task::Poll,
    time::{Duration, Instant},
};

use roplat_behavior::behavior::{Arm, EndPoint, FlangeSpace, JointSpace, Joints};
use roplat_behavior::{
    ArmState, ArmTorqueControl, AsyncControlCallback, AsyncControlWith, ControlObservation,
    ControlStep, ControlWith, JointPositionControl, JointState, JointVelocityControl, LoadState,
    MoveTo, Pose, Robot, RobotException, RobotResult, TorqueControl,
};
use roplat_trajectory::path_generate;

use crate::rsbullet_robot::{RsBulletControlSpace, Session, clamp_position};
use crate::{RsBullet, RsBulletRobot, types::RsBulletRobotState};

/// A simulated device: one robot driving its own RsBullet world.
///
/// A real arm paces its control session itself. An [`RsBulletRobot`] cannot,
/// because its world only advances in the engine's `step`. This type owns the
/// world together with the robot, so a [`ControlWith`] or [`AsyncControlWith`]
/// session steps it once per control cycle, and a graph written against a real
/// driver, such as a `ControlRhythm` or `AsyncControlRhythm` over
/// `JointPositionControl<N>`, runs unchanged in simulation.
///
/// Each cycle observes the native state, calls the controller with the physics
/// timestep, applies the command and integrates once. `Break` applies nothing;
/// `done` applies its command and integrates once more. Other bodies keep their
/// queued controls, which run inside every integration. However a session ends,
/// the joints are left holding their measured position.
///
/// Sessions run in logical time unless [`RsBulletDevice::realtime`] is set;
/// asynchronous sessions yield to sibling futures after every integration.
/// Nodes that need the simulated time read [`RsBullet::clock`].
pub struct RsBulletDevice<R> {
    engine: RsBullet,
    robot: RsBulletRobot<R>,
    realtime: bool,
}

impl<R> RsBulletDevice<R> {
    /// Fails unless `robot` is a valid handle loaded into `engine`.
    pub fn new(engine: RsBullet, robot: RsBulletRobot<R>) -> RobotResult<Self> {
        if !robot.is_valid() || !robot.belongs_to(&engine) {
            return Err(RobotException::CommandException(
                "a simulated device needs a valid robot loaded into its own world".into(),
            ));
        }
        Ok(Self {
            engine,
            robot,
            realtime: false,
        })
    }

    /// Pace control cycles at the physics timestep on the wall clock. A late
    /// cycle starts at once; no integration is skipped. Only blocking sessions
    /// can wait; an asynchronous session then reports unsupported.
    pub fn realtime(mut self, enabled: bool) -> Self {
        self.realtime = enabled;
        self
    }

    pub fn engine(&self) -> &RsBullet {
        &self.engine
    }

    /// Resetting the world through this invalidates the robot handle.
    pub fn engine_mut(&mut self) -> &mut RsBullet {
        &mut self.engine
    }

    pub fn robot(&self) -> &RsBulletRobot<R> {
        &self.robot
    }

    pub fn robot_mut(&mut self) -> &mut RsBulletRobot<R> {
        &mut self.robot
    }

    pub fn into_parts(self) -> (RsBullet, RsBulletRobot<R>) {
        (self.engine, self.robot)
    }

    fn session<S, F>(&mut self, mut controller: F) -> RobotResult<()>
    where
        S: RsBulletControlSpace<R>,
        F: FnMut(S::Obs, Duration) -> ControlStep<S::Command>,
    {
        let realtime = self.realtime;
        let mut session = Session::<R, S>::begin(&self.robot, &mut self.engine)?;
        let period = session.period();
        let mut next_cycle = Instant::now();
        let mut cycles = || -> RobotResult<()> {
            loop {
                let obs = session.observe()?;
                if !session.conclude(controller(obs, period))? {
                    return Ok(());
                }
                if realtime {
                    next_cycle += period;
                    if let Some(wait) = next_cycle.checked_duration_since(Instant::now()) {
                        std::thread::sleep(wait);
                    }
                }
            }
        };
        let outcome = cycles();
        session.end(outcome)
    }

    async fn session_async<S, C>(&mut self, callback: &mut C) -> RobotResult<()>
    where
        S: RsBulletControlSpace<R>,
        C: AsyncControlCallback<S::Obs, S::Command>,
    {
        if self.realtime {
            return Err(RobotException::UnprocessableInstructionError(
                "realtime pacing needs a blocking ControlWith session; async sessions run in logical time".into(),
            ));
        }
        let mut session = Session::<R, S>::begin(&self.robot, &mut self.engine)?;
        let period = session.period();
        let outcome = async {
            loop {
                let obs = session.observe()?;
                if !session.conclude(callback.call(obs, period).await)? {
                    return Ok(());
                }
                yield_now().await;
            }
        }
        .await;
        session.end(outcome)
    }
}

/// Lets sibling futures run between cycles of a logical-time session.
async fn yield_now() {
    let mut yielded = false;
    std::future::poll_fn(|cx| {
        if yielded {
            Poll::Ready(())
        } else {
            yielded = true;
            cx.waker().wake_by_ref();
            Poll::Pending
        }
    })
    .await
}

impl<R> Robot for RsBulletDevice<R> {
    type State = RsBulletRobotState;
    const CONTROL_PERIOD: f64 = <RsBulletRobot<R> as Robot>::CONTROL_PERIOD;

    fn version() -> String {
        "RsBulletDevice".to_string()
    }

    fn read_state(&mut self) -> RobotResult<Self::State> {
        self.robot.read_state()
    }
}

impl<R> ControlObservation for RsBulletDevice<R> {
    fn before<H>(&mut self, observer: H) -> &mut Self
    where
        H: FnMut(&Self::State, Duration) + Send + 'static,
    {
        self.robot.before(observer);
        self
    }

    fn after<H>(&mut self, observer: H) -> &mut Self
    where
        H: FnMut(&Self::State, Duration) + Send + 'static,
    {
        self.robot.after(observer);
        self
    }
}

fn hold_joint_position<const N: usize>(state: &JointState<N>) -> [f64; N] {
    state
        .cmd
        .q()
        .or(state.des.q())
        .or(state.meas.q())
        .unwrap_or([0.0; N])
}

fn hold_joint_velocity<const N: usize>(_state: &JointState<N>) -> [f64; N] {
    [0.0; N]
}

fn hold_joint_torque<const N: usize>(state: &JointState<N>) -> [f64; N] {
    state
        .cmd
        .tau()
        .or(state.des.tau())
        .or(state.meas.tau())
        .unwrap_or([0.0; N])
}

fn hold_arm_torque<const N: usize>(state: &ArmState<N>) -> [f64; N] {
    hold_joint_torque(&state.joint)
}

macro_rules! control_channel {
    ($channel:ident, $obs:ident, $hold:ident) => {
        impl<const N: usize, R> ControlWith<$channel<N>> for RsBulletDevice<R>
        where
            R: Joints<N>,
        {
            fn hold_command(state: &$obs<N>) -> [f64; N] {
                $hold(state)
            }

            fn control_with_flow<F>(&mut self, closure: F) -> RobotResult<()>
            where
                F: FnMut($obs<N>, Duration) -> ControlStep<[f64; N]>,
            {
                self.session::<$channel<N>, F>(closure)
            }
        }

        impl<const N: usize, R> AsyncControlWith<$channel<N>> for RsBulletDevice<R>
        where
            R: Joints<N> + Send + Sync,
        {
            fn control_native_async<C>(
                &mut self,
                callback: &mut C,
            ) -> impl Future<Output = RobotResult<()>> + Send
            where
                C: AsyncControlCallback<$obs<N>, [f64; N]>,
            {
                self.session_async::<$channel<N>, C>(callback)
            }
        }
    };
}

control_channel!(JointPositionControl, JointState, hold_joint_position);
control_channel!(JointVelocityControl, JointState, hold_joint_velocity);
control_channel!(TorqueControl, JointState, hold_joint_torque);
control_channel!(ArmTorqueControl, ArmState, hold_arm_torque);

impl<const N: usize, R> MoveTo<JointSpace<N>> for RsBulletDevice<R>
where
    R: Joints<N>,
{
    /// Returns once the S-curve to `target` has been integrated to its end,
    /// like a real arm's motion, unlike the queued [`RsBulletRobot`] motion.
    fn move_to(&mut self, target: [f64; N]) -> RobotResult<()> {
        let target = clamp_position(target, R::JOINT_MIN, R::JOINT_MAX)?;
        let mut path = None;
        let mut elapsed = Duration::ZERO;
        let mut unobserved = false;
        <Self as ControlWith<JointPositionControl<N>>>::control_with_flow(self, |state, dt| {
            if path.is_none() {
                let Some(start) = state.meas.q() else {
                    unobserved = true;
                    return ControlFlow::Break(());
                };
                path = Some(path_generate::joint_s_curve(
                    &start,
                    &target,
                    &R::JOINT_VEL_BOUND,
                    &R::JOINT_ACC_BOUND,
                    &R::JOINT_JERK_BOUND,
                ));
            }
            let (generate, total) = path.as_ref().expect("path planned on the first cycle");
            // A command applies over the next integration; aim at its end.
            elapsed += dt;
            ControlFlow::Continue((generate(elapsed), elapsed >= *total))
        })?;
        if unobserved {
            return Err(RobotException::CommandException(
                "joint positions are unavailable; the motion was not started".into(),
            ));
        }
        Ok(())
    }
}

impl<R> MoveTo<FlangeSpace> for RsBulletDevice<R>
where
    R: EndPoint,
{
    fn move_to(&mut self, _target: Pose) -> RobotResult<()> {
        Err(RobotException::UnprocessableInstructionError(
            "RsBulletDevice does not plan cartesian motions; use JointSpace with native IK".into(),
        ))
    }
}

impl<const N: usize, R> Joints<N> for RsBulletDevice<R>
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

impl<R> EndPoint for RsBulletDevice<R>
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

/// The `with_*` setters are no-ops, as on [`RsBulletRobot`]: limits come from
/// the robot description and the model.
impl<const N: usize, R> Arm<N> for RsBulletDevice<R>
where
    R: Joints<N> + EndPoint,
{
    fn state(&mut self) -> RobotResult<ArmState<N>> {
        Arm::<N>::state(&mut self.robot)
    }

    fn set_load(&mut self, load: LoadState) -> RobotResult<()> {
        Arm::<N>::set_load(&mut self.robot, load)
    }

    fn get_joint(&self) -> [f64; N] {
        Arm::<N>::get_joint(&self.robot)
    }

    fn get_endpoint(&self) -> Pose {
        Arm::<N>::get_endpoint(&self.robot)
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
