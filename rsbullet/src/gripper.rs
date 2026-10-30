//! A simulated parallel gripper behind the shared [`Gripper`] contract.
//!
//! [`Gripper`]: roplat_behavior::Gripper

use std::time::Instant;

use roplat_behavior::{
    RobotException, RobotResult,
    msgs::{GripperCommand, GripperState},
};
use roplat_world::PhysicsEngine;
use rsbullet_core::{
    BulletResult, ConstraintCreateOptions, ControlModeArray, JointType, PhysicsClient,
};

use crate::RsBullet;

/// A grasp counts as held while the opening stays within this distance (m) of
/// the commanded width, libfranka's default grasp tolerance.
pub const GRASP_EPSILON: f64 = 0.01;
/// A move succeeds when it ends within this distance (m) of its width.
pub const MOVE_TOLERANCE: f64 = 0.002;
/// Opening speed (m/s) of `home_gripper`.
pub const HOMING_SPEED: f64 = 0.1;

/// Finger speed (m/s) below which the fingers count as still.
const SETTLED_SPEED: f64 = 1e-3;
/// How long (s) the fingers must stay still for a motion to end.
const SETTLE_TIME: f64 = 0.05;
/// How long (s) a motion may take to settle after its ramp.
const SETTLE_LIMIT: f64 = 2.0;
/// Distance (m) between a finger and its target beyond which a still finger
/// is blocked rather than arrived.
const STALL_GAP: f64 = 2e-3;
/// Force bound of the finger coupling, as a multiple of the effort limit.
const COUPLING_STRENGTH: f64 = 5.0;

/// Two prismatic finger joints of one body, driven as a parallel gripper.
///
/// Each finger closes at zero and opens along its axis, as in the Panda
/// hand, so the opening is the sum of the two finger positions and each
/// finger travels half of a width change.
///
/// A gear constraint keeps the fingers mirrored, as one drive moves both
/// fingers of a parallel hand (pybullet's Panda example couples them the
/// same way). Without it, two force-limited fingers squeezing an object
/// resist its sideways push with nothing, and a held object slides along
/// the closing axis whenever the arm accelerates.
pub(crate) struct ParallelFingers {
    body: i32,
    joints: [i32; 2],
    stroke: f64,
    max_force: f64,
    /// The width of a successful grasp that is still being squeezed.
    grasp: Option<f64>,
}

impl ParallelFingers {
    /// Resolves the fingers by joint name. They must be prismatic, outside
    /// the device's controlled joints, close at zero, share one finite
    /// stroke and carry a finite URDF effort limit.
    pub(crate) fn attach(
        client: &mut PhysicsClient,
        body: i32,
        controlled: &[i32],
        names: [&str; 2],
    ) -> RobotResult<Self> {
        if names[0] == names[1] {
            return Err(invalid("the two fingers must be different joints".into()));
        }
        let infos = (0..client.get_num_joints(body))
            .map(|joint| client.get_joint_info(body, joint))
            .collect::<BulletResult<Vec<_>>>()?;
        let mut fingers = Vec::with_capacity(2);
        for name in names {
            let info = infos
                .iter()
                .find(|info| info.joint_name == name)
                .ok_or_else(|| invalid(format!("finger joint {name:?} is not in the model")))?;
            let problem = if info.joint_type != JointType::Prismatic {
                Some("is not prismatic")
            } else if controlled.contains(&info.joint_index) {
                Some("is one of the device's controlled joints")
            } else if info.lower_limit != 0.0
                || !(info.upper_limit.is_finite() && info.upper_limit > 0.0)
            {
                Some("must close at zero and open to a finite positive limit")
            } else if !(info.max_force.is_finite() && info.max_force > 0.0) {
                Some("needs a finite positive URDF effort limit")
            } else {
                None
            };
            if let Some(problem) = problem {
                return Err(invalid(format!("finger joint {name:?} {problem}")));
            }
            fingers.push(info);
        }
        let [first, second] = [fingers[0], fingers[1]];
        if first.upper_limit != second.upper_limit {
            return Err(invalid("the two fingers must share one stroke".into()));
        }
        let max_force = first.max_force.min(second.max_force);
        client.create_constraint(&ConstraintCreateOptions {
            parent_body: body,
            parent_link: Some(first.joint_index as usize),
            child_body: body,
            child_link: Some(second.joint_index as usize),
            joint_type: JointType::Gear,
            joint_axis: [1.0, 0.0, 0.0],
            max_applied_force: COUPLING_STRENGTH * max_force,
            gear_ratio: Some(-1.0),
            erp: Some(0.1),
            ..Default::default()
        })?;
        Ok(Self {
            body,
            joints: [first.joint_index, second.joint_index],
            stroke: first.upper_limit,
            max_force,
            grasp: None,
        })
    }

    fn max_width(&self) -> f64 {
        2.0 * self.stroke
    }

    /// Position and velocity of each finger.
    fn fingers(&self, client: &mut PhysicsClient) -> RobotResult<[(f64, f64); 2]> {
        let states = client.get_joint_states(self.body, &self.joints)?;
        match states.as_slice() {
            [first, second] => Ok([
                (first.position, first.velocity),
                (second.position, second.velocity),
            ]),
            _ => Err(RobotException::CommandException(
                "incomplete finger state".into(),
            )),
        }
    }

    pub(crate) fn read(&self, client: &mut PhysicsClient) -> RobotResult<GripperState> {
        let [(first, _), (second, _)] = self.fingers(client)?;
        let width = first + second;
        let grasped = self
            .grasp
            .is_some_and(|grasp| (width - grasp).abs() <= GRASP_EPSILON);
        Ok(GripperState {
            width,
            max_width: self.max_width(),
            grasped: grasped as u32,
            reserved: 0,
        })
    }

    /// Moves or grasps, stepping `engine` until the fingers come to rest.
    ///
    /// A move drives both fingers to half the width with the fingers' effort
    /// limit. A grasp closes (or, for a width beyond the current opening,
    /// opens) all the way with `force`, so an object between the fingers
    /// stops them; it succeeds when they stop within [`GRASP_EPSILON`] of
    /// the width, and then keeps squeezing.
    pub(crate) fn command(
        &mut self,
        engine: &mut RsBullet,
        realtime: bool,
        command: GripperCommand,
    ) -> RobotResult<()> {
        let GripperCommand { width, speed, force } = command;
        let width = up_to(width, self.max_width()).ok_or_else(|| {
            invalid(format!(
                "gripper width {width} is outside [0, {}]",
                self.max_width()
            ))
        })?;
        if !(speed.is_finite() && speed > 0.0) {
            return Err(invalid(format!("gripper speed {speed} must be positive")));
        }
        let force = up_to(force, self.max_force).ok_or_else(|| {
            invalid(format!(
                "gripper force {force} is outside [0, {}]",
                self.max_force
            ))
        })?;
        self.grasp = None;
        let [(first, _), (second, _)] = self.fingers(engine.client_mut())?;
        let grasping = force > 0.0;
        let goal = if !grasping {
            width / 2.0
        } else if width <= first + second {
            0.0
        } else {
            self.stroke
        };
        let effort = if grasping { force } else { self.max_force };
        let reached = match self
            .drive(engine, realtime, [first, second], goal, speed / 2.0, effort)
            .and_then(|()| self.read(engine.client_mut()))
        {
            Ok(state) => state.width,
            Err(error) => {
                // As after a control session: hold unless the world faulted.
                if !engine.is_faulted() {
                    let _ = self.hold(engine.client_mut());
                }
                return Err(error);
            }
        };
        let client = engine.client_mut();
        if grasping {
            if (reached - width).abs() <= GRASP_EPSILON {
                self.grasp = Some(width);
                self.set_targets(client, [goal; 2], force)?;
            } else {
                self.hold(client)?;
            }
            Ok(())
        } else if (reached - width).abs() <= MOVE_TOLERANCE {
            Ok(())
        } else {
            self.hold(client)?;
            Err(RobotException::CommandException(format!(
                "gripper: the move to {width} m stopped at {reached:.4} m"
            )))
        }
    }

    pub(crate) fn home(&mut self, engine: &mut RsBullet, realtime: bool) -> RobotResult<()> {
        let open = GripperCommand { width: self.max_width(), speed: HOMING_SPEED, force: 0.0 };
        self.command(engine, realtime, open)
    }

    /// Holds the measured finger positions, releasing any grasp force.
    pub(crate) fn hold(&mut self, client: &mut PhysicsClient) -> RobotResult<()> {
        self.grasp = None;
        let [(first, _), (second, _)] = self.fingers(client)?;
        self.set_targets(client, [first, second], self.max_force)
    }

    /// Ramps both finger targets from `start` to `goal` at `finger_speed`,
    /// integrating once per target. The motion ends once the fingers have
    /// been still for [`SETTLE_TIME`], either after the ramp or earlier when
    /// something blocks them.
    fn drive(
        &self,
        engine: &mut RsBullet,
        realtime: bool,
        start: [f64; 2],
        goal: f64,
        finger_speed: f64,
        effort: f64,
    ) -> RobotResult<()> {
        let period = engine.step_time();
        let dt = period.as_secs_f64();
        let travel = (goal - start[0]).abs().max((goal - start[1]).abs());
        let ramp = (travel / (finger_speed * dt)).ceil() as u64;
        let settle = (SETTLE_TIME / dt).ceil() as u64;
        let limit = ramp + (SETTLE_LIMIT / dt).ceil() as u64;
        let mut still = 0;
        let mut next_cycle = Instant::now();
        for step in 1..=limit {
            let progress = if ramp == 0 {
                1.0
            } else {
                (step as f64 / ramp as f64).min(1.0)
            };
            let target = start.map(|from| from + (goal - from) * progress);
            self.set_targets(engine.client_mut(), target, effort)?;
            engine.step()?;
            let fingers = self.fingers(engine.client_mut())?;
            if fingers
                .iter()
                .all(|&(_, velocity)| velocity.abs() < SETTLED_SPEED)
            {
                still += 1;
            } else {
                still = 0;
            }
            let blocked = (0..2).any(|i| (target[i] - fingers[i].0).abs() > STALL_GAP);
            if still >= settle && (step >= ramp || blocked) {
                return Ok(());
            }
            if realtime {
                next_cycle += period;
                if let Some(wait) = next_cycle.checked_duration_since(Instant::now()) {
                    std::thread::sleep(wait);
                }
            }
        }
        Err(RobotException::CommandException(
            "gripper: the fingers did not come to rest".into(),
        ))
    }

    fn set_targets(
        &self,
        client: &mut PhysicsClient,
        target: [f64; 2],
        effort: f64,
    ) -> RobotResult<()> {
        client.set_joint_motor_control_array(
            self.body,
            &self.joints,
            ControlModeArray::Position(&target),
            Some(&[effort; 2]),
        )?;
        Ok(())
    }
}

/// `value` within `[0, limit]`, clamped to `limit` when it exceeds it only by
/// rounding: Bullet keeps joint limits in single precision, so a 0.04 m
/// stroke reads back as 0.0399999991.
fn up_to(value: f64, limit: f64) -> Option<f64> {
    let slack = limit * f64::from(f32::EPSILON);
    (0.0..=limit + slack)
        .contains(&value)
        .then(|| value.min(limit))
}

fn invalid(message: String) -> RobotException {
    RobotException::UnprocessableInstructionError(message)
}
