//! Task 3: a parameter sweep.
//!
//! The same out-and-back motion, from `HOME` to above the cube's spot and
//! back, runs three times at each of three speeds (the fraction of the joint
//! velocity, acceleration and jerk limits COPP times the path to). Each run
//! reports the timed duration and the largest tracking error.
//!
//! Structure: `speeds` (a collecting iterator rhythm) yields one speed per
//! cycle; its domain creates `repeats` (another collecting rhythm) afresh,
//! whose cycles inherit the speed. Each rhythm returns its cycles' results.
use std::sync::Arc;

use roplat::RoplatResult;
use roplat::rhythm::IntoRhythm;
use roplat_behavior::msgs::ManualClock;

use crate::act::act;
use crate::world::{Device, HOME, Motion, Planner, move_to};

/// The speeds swept.
pub const SPEEDS: [f64; 3] = [0.15, 0.3, 0.6];
/// Runs at each speed.
pub const REPEATS: usize = 3;

/// One out-and-back run.
#[derive(Clone, Copy, Debug)]
pub struct Run {
    pub speed: f64,
    pub out: Motion,
    pub back: Motion,
}

#[roplat::system]
pub async fn sweep(
    mut arm: Device,
    mut planner: Planner,
    clock: Arc<ManualClock>,
    target: [f64; 7],
) -> RoplatResult<(Device, Planner, Vec<Vec<Run>>)> {
    let mut speeds = SPEEDS.into_iter().into_collect_rhythm::<Vec<Run>>();
    let out_clock = clock.clone();
    let mut out = act("out", move |arm, planner, speed: f64| {
        let motion = move_to(arm, planner, &out_clock, target, speed)?;
        Ok((speed, motion))
    });
    let mut back = act("back", move |arm, planner, (speed, out): (f64, Motion)| {
        let motion = move_to(arm, planner, &clock, HOME, speed)?;
        Ok(Run {
            speed,
            out,
            back: motion,
        })
    });

    speeds
        >> |speed| {
            let mut repeats = (0..REPEATS).into_collect_rhythm::<Run>();
            repeats
                >> |_repeat| {
                    &mut arm >> out.resources.0;
                    &mut planner >> out.resources.1;
                    &mut arm >> back.resources.0;
                    &mut planner >> back.resources.1;
                    speed >> out;
                    out? >> back;
                    back?
                };
            repeats
        };

    Ok((arm, planner, speeds.output))
}
