//! Task 4: stopping a motion on contact.
//!
//! The open hand moves straight down over the table on a trajectory that
//! ends 2 cm inside it, as if the table's height were not known. The control
//! loop runs in the graph: each cycle the follower gives a command, and the
//! probe compares the measured joint torques with those at the start of the
//! descent. When they differ by more than a threshold the probe ends the
//! session with `done`, holding where the arm is, and the task backs off.
//!
//! What it shows about interruption: the condition is checked inside the
//! running control session, once per cycle, and the session ends because the
//! domain says so. Nothing outside the session can cut it short; the steps of
//! the other tasks, which run their sessions inside a node's `process`,
//! cannot be interrupted at all until they return.
use std::{sync::Arc, time::Duration};

use roplat::{Execution, Lifecycle, Node, RoplatError, RoplatResult};
use roplat_behavior::JointPositionControl;
use roplat_behavior::msgs::{JointState, ManualClock};
use roplat_behavior::roplat::ControlRhythm;

use crate::act::{Failure, act};
use crate::world::{Device, FINGER_LINKS, Follower, Planner, SPEED, joints, move_to};

/// The descent's speed: the fraction of the joint limits COPP times it to.
pub const DESCENT_SPEED: f64 = 0.05;
/// The joint torque change (N·m) taken as contact. In free descent the
/// change stays below about 4.2 N·m (joint 2, while accelerating); the first
/// cycle in contact raises it to about 30 N·m.
pub const THRESHOLD: f64 = 10.0;

/// When the descent stopped.
#[derive(Clone, Copy, Debug)]
pub struct Touch {
    /// Time into the descent.
    pub after: Duration,
    /// The joint positions held.
    pub q: [f64; 7],
    /// The largest joint torque change (N·m) when the probe stopped.
    pub torque: f64,
}

/// Ends the control session when a joint torque leaves the value it had when
/// the session began by more than `threshold` (N·m).
pub struct Probe {
    threshold: f64,
    baseline: Option<[f64; 7]>,
    elapsed: Duration,
    /// The largest change seen before the probe stopped the session.
    pub quiet: f64,
    pub touch: Option<Touch>,
}

impl Probe {
    pub fn new(threshold: f64) -> Self {
        Self {
            threshold,
            baseline: None,
            elapsed: Duration::ZERO,
            quiet: 0.0,
            touch: None,
        }
    }
}

impl Lifecycle for Probe {
    type Error = RoplatError;
}

impl Node for Probe {
    type Resources = ();
    type Input = ((JointState<7>, Duration), ([f64; 7], bool));
    type Output = Result<([f64; 7], bool), Failure>;

    async fn process(
        &mut self,
        ((state, period), (command, done)): Self::Input,
        (): (),
    ) -> Self::Output {
        let missing = || Failure::Device("no measured joint torques or positions".into());
        let (Some(tau), Some(q)) = (state.meas.tau(), state.meas.q()) else {
            return Err(missing());
        };
        self.elapsed += period;
        let baseline = *self.baseline.get_or_insert(tau);
        let change = (0..7)
            .map(|i| (tau[i] - baseline[i]).abs())
            .fold(0.0, f64::max);
        if change > self.threshold {
            self.touch = Some(Touch {
                after: self.elapsed,
                q,
                torque: change,
            });
            return Ok((q, true));
        }
        self.quiet = self.quiet.max(change);
        Ok((command, done))
    }
}

/// Lets the fingers touch the table, as they do when the probe stops them.
fn rest_on_table(planner: &mut Planner, allowed: bool) {
    for finger in FINGER_LINKS {
        if allowed {
            planner.scene.allowed_mut().allow("table", finger);
        } else {
            planner.scene.allowed_mut().forbid("table", finger);
        }
    }
}

#[roplat::system]
pub async fn contact(
    mut arm: Device,
    mut planner: Planner,
    clock: Arc<ManualClock>,
    mut descent: Follower,
    above: [f64; 7],
    threshold: f64,
) -> RoplatResult<(Device, Planner, Option<Touch>, f64)> {
    let mut control = ControlRhythm::<Device, JointPositionControl<7>>::new();
    let mut probe = Probe::new(threshold);
    let reach_clock = clock.clone();
    let mut reach = act("move above the table", move |arm, planner, (): ()| {
        move_to(arm, planner, &reach_clock, above, SPEED)?;
        Ok(())
    });
    let mut back_off = act("back off", move |arm, planner, outcome: Execution<()>| {
        Failure::from_outcome(outcome)?;
        let q = joints(arm)?;
        let up = nalgebra::Isometry3::translation(0.0, 0.0, 0.10) * planner.tip(&q)?;
        let goal = planner.reach_pose(up, q)?;
        rest_on_table(planner, true);
        let moved = move_to(arm, planner, &clock, goal, SPEED);
        rest_on_table(planner, false);
        moved.map(|_| ())
    });

    &mut arm >> reach.resources.0;
    &mut planner >> reach.resources.1;
    &mut arm >> control.resources.0;
    &mut arm >> back_off.resources.0;
    &mut planner >> back_off.resources.1;
    reach? >> control.trigger;
    control
        >> |sample| {
            sample >> descent;
            sample >> probe.input.0;
            descent? >> probe.input.1;
            probe?
        };
    control.outcome >> back_off;
    back_off?;

    Ok((arm, planner, probe.touch, probe.quiet))
}
