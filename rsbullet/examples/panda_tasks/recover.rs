//! Task 1: pick and place with recovery.
//!
//! A short post stands 3 cm beside the cube. Two ways of holding the cube are
//! tried in turn: fingers closing along y would touch the post, so that way
//! fails at planning and the task falls back to fingers closing along x.
//! Perception is wrong once (8 cm off, injected on its second reading), so
//! the first grasp of that way closes on air; the miss is retried, perception
//! reads again and the second grasp holds. The cube is then carried over the
//! wall and placed.
//!
//! Structure: `ways` (a retry whose attempt picks the way) wraps `tries` (a
//! retry of a missed grasp), which wraps `attempt` (a `once` group whose
//! steps raise failures with `?`). `.outcome` turns the group's failure back
//! into data, and `judge` decides whether trying again can help.
use std::sync::Arc;

use nalgebra::Isometry3;
use roplat::rhythm::{once, retry};
use roplat::{Execution, RoplatResult};
use roplat_behavior::msgs::ManualClock;

use crate::act::Failure;
use crate::steps::{self, Grasp};
use crate::world::{Block, Cell, Device, Planner, TABLE_TOP, Yaw};

/// The cube's resting spot.
pub const CUBE_AT: [f64; 2] = [0.45, -0.15];
/// A post 3 cm from the cube's face along -y, 6 cm high.
pub const POST: Block = Block {
    name: "post",
    half: [0.02, 0.01, 0.03],
    at: [0.45, -0.21, TABLE_TOP + 0.03],
};
/// Where the cube goes, beyond the wall.
pub const PLACE: [f64; 2] = [0.45, 0.20];
/// The ways of holding the cube, in the order tried.
const WAYS: [Yaw; 2] = [Yaw::ALONG_Y, Yaw::ALONG_X];
/// The error perception makes on each reading (m, x and y).
const PERCEPTION_ERRORS: [[f64; 2]; 2] = [[0.0, 0.0], [0.08, 0.0]];

/// What the task did.
#[derive(Debug)]
pub struct Report {
    pub ways_tried: usize,
    pub grasps_tried: usize,
    pub placed: Isometry3<f64>,
}

/// A grasp is retried only when it missed; any other failure ends the
/// attempts with that failure, and the next way is tried.
fn classify(outcome: Execution<Grasp>) -> Result<Result<Grasp, Failure>, Failure> {
    match Failure::from_outcome(outcome) {
        Err(failure) if failure.retryable() => Err(failure),
        result => Ok(result),
    }
}

#[roplat::system]
pub async fn recover(
    mut arm: Device,
    mut planner: Planner,
    cube: i32,
    clock: Arc<ManualClock>,
) -> RoplatResult<(Device, Planner, Report)> {
    let mut ways = retry::<(), Grasp, Failure>(WAYS.len());
    let mut tries = retry::<Yaw, Result<Grasp, Failure>, Failure>(3);
    let mut attempt = once::<Yaw, Grasp>();

    let mut choose = roplat::node::map(|((), way): ((), usize)| WAYS[way]);
    let mut judge = roplat::node::map(classify);
    let mut flatten =
        roplat::node::map(|tried: Result<Result<Grasp, Failure>, Failure>| tried.and_then(|g| g));

    let mut open = steps::open(&clock);
    let mut perceive = steps::perceive(cube, &PERCEPTION_ERRORS);
    let mut plan = steps::plan_grasp(PLACE);
    let mut approach = steps::approach(&clock);
    let mut descend = steps::descend(&clock);
    let mut close = steps::close(cube);
    let mut lift = steps::lift(&clock);
    let mut carry = steps::carry(&clock);
    let mut lower = steps::lower(&clock);
    let mut release = steps::release();
    let mut retreat = steps::retreat(&clock);

    ways >> |way| {
        way >> choose;
        choose
            >> tries
            >> |(yaw, _try)| {
                yaw >> attempt
                    >> |yaw| {
                        &mut arm >> open.resources.0;
                        &mut planner >> open.resources.1;
                        &mut arm >> perceive.resources.0;
                        &mut planner >> perceive.resources.1;
                        &mut arm >> plan.resources.0;
                        &mut planner >> plan.resources.1;
                        &mut arm >> approach.resources.0;
                        &mut planner >> approach.resources.1;
                        &mut arm >> descend.resources.0;
                        &mut planner >> descend.resources.1;
                        &mut arm >> close.resources.0;
                        &mut planner >> close.resources.1;
                        open? >> perceive;
                        perceive? >> plan.input.0;
                        yaw >> plan.input.1;
                        plan? >> approach;
                        approach? >> descend;
                        descend? >> close;
                        close?
                    };
                attempt.outcome >> judge;
                judge
            };
        tries >> flatten;
        flatten
    };

    &mut arm >> lift.resources.0;
    &mut planner >> lift.resources.1;
    &mut arm >> carry.resources.0;
    &mut planner >> carry.resources.1;
    &mut arm >> lower.resources.0;
    &mut planner >> lower.resources.1;
    &mut arm >> release.resources.0;
    &mut planner >> release.resources.1;
    &mut arm >> retreat.resources.0;
    &mut planner >> retreat.resources.1;
    ways? >> lift;
    lift? >> carry;
    carry? >> lower;
    lower? >> release;
    release? >> retreat;
    retreat?;

    let placed = Cell::cube_pose(&mut arm, cube)?;
    let report = Report {
        ways_tried: ways.attempts(),
        grasps_tried: tries.attempts(),
        placed,
    };
    Ok((arm, planner, report))
}
