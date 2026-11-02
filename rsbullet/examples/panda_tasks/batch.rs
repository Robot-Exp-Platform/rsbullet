//! Task 2: a batch of repeated trials.
//!
//! The same pick and place runs twice from each of three spots. Between
//! trials the simulator resets the cell, as an experimenter would. The third
//! spot, at the table's far corner, is beyond the arm's reach, so its trials
//! fail at planning; each failure is recorded and the batch goes on.
//!
//! Structure: `trials` (a collecting iterator rhythm) runs one trial per
//! cycle and returns every cycle's record; each trial's steps are a `once`
//! group whose `.outcome` is recorded instead of ending the batch.
use std::sync::Arc;

use roplat::rhythm::{IntoRhythm, once};
use roplat::{Execution, RoplatResult};
use roplat_behavior::msgs::ManualClock;

use crate::act::Failure;
use crate::steps::{self, Placement};
use crate::world::{Device, Planner, Yaw};

/// Where the cube rests for each trial: two spots within reach and one at the
/// table's far corner, 0.93 m from the arm's base.
pub const SPOTS: [[f64; 2]; 3] = [[0.45, -0.15], [0.55, -0.25], [0.85, -0.38]];
/// Trials from each spot.
pub const REPEATS: usize = 2;
/// Where the cube goes, beyond the wall.
pub const PLACE: [f64; 2] = [0.45, 0.20];

/// One trial's record.
#[derive(Debug)]
pub struct Trial {
    pub spot: [f64; 2],
    pub repeat: usize,
    pub result: Result<Placement, Failure>,
}

impl Trial {
    /// Whether the cube ended on the table within 5 mm of its place.
    pub fn succeeded(&self) -> bool {
        matches!(&self.result, Ok(placement) if placement.on_table && placement.off < 0.005)
    }
}

fn record((outcome, spot, repeat): (Execution<Placement>, [f64; 2], usize)) -> Trial {
    Trial {
        spot,
        repeat,
        result: Failure::from_outcome(outcome),
    }
}

#[roplat::system]
pub async fn batch(
    mut arm: Device,
    mut planner: Planner,
    cube: i32,
    clock: Arc<ManualClock>,
    body: i32,
    arm_joints: [i32; 7],
) -> RoplatResult<(Device, Planner, Vec<Trial>)> {
    let plan: Vec<([f64; 2], usize)> = SPOTS
        .iter()
        .flat_map(|&spot| (0..REPEATS).map(move |repeat| (spot, repeat)))
        .collect();
    let mut trials = plan.into_iter().into_collect_rhythm::<Trial>();
    let mut run = once::<f64, Placement>();
    let mut record = roplat::node::map(record);
    let yaw = Yaw::ALONG_Y;

    let mut reset = steps::reset(cube, body, arm_joints);
    let mut perceive = steps::perceive(cube, &[]);
    let mut grasp = steps::plan_grasp(PLACE);
    let mut approach = steps::approach(&clock);
    let mut descend = steps::descend(&clock);
    let mut close = steps::close(cube);
    let mut lift = steps::lift(&clock);
    let mut carry = steps::carry(&clock);
    let mut lower = steps::lower(&clock);
    let mut release = steps::release();
    let mut retreat = steps::retreat(&clock);
    let mut measure = steps::measure(cube, PLACE);

    trials
        >> |(spot, repeat)| {
            &mut arm >> reset.resources.0;
            &mut planner >> reset.resources.1;
            spot >> reset;
            reset?
                >> run
                >> |start| {
                    &mut arm >> perceive.resources.0;
                    &mut planner >> perceive.resources.1;
                    &mut arm >> grasp.resources.0;
                    &mut planner >> grasp.resources.1;
                    &mut arm >> approach.resources.0;
                    &mut planner >> approach.resources.1;
                    &mut arm >> descend.resources.0;
                    &mut planner >> descend.resources.1;
                    &mut arm >> close.resources.0;
                    &mut planner >> close.resources.1;
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
                    &mut arm >> measure.resources.0;
                    &mut planner >> measure.resources.1;
                    perceive? >> grasp.input.0;
                    &yaw >> grasp.input.1;
                    grasp? >> approach;
                    approach? >> descend;
                    descend? >> close;
                    close? >> lift;
                    lift? >> carry;
                    carry? >> lower;
                    lower? >> release;
                    release? >> retreat;
                    retreat? >> measure.input.0;
                    start >> measure.input.1;
                    measure?
                };
            run.outcome >> record.input.0;
            spot >> record.input.1;
            repeat >> record.input.2;
            record
        };

    Ok((arm, planner, trials.output))
}
