//! The steps of a pick and place, as nodes the tasks wire together.
//!
//! Each builder returns a fresh node; a graph calls each node in one place,
//! so a task that does a step twice builds it twice.
use std::sync::{Arc, Mutex};

use nalgebra::Isometry3;
use roplat_behavior::Gripper;
use roplat_behavior::msgs::ManualClock;
use roplat_scene::{CollisionCheck, SceneError};

use crate::act::{Act, Failure, act};
use crate::world::{
    ABOVE, Cell, Device, FINGER_LINKS, OPEN, Planner, REST, SPEED, Yaw, grip, joints, move_to,
};

/// Configurations for one way of holding the cube and setting it down.
#[derive(Clone, Debug)]
pub struct Grasp {
    pub above: [f64; 7],
    pub at: [f64; 7],
    pub above_place: [f64; 7],
    pub at_place: [f64; 7],
}

fn scene(error: SceneError) -> Failure {
    Failure::Plan(error.to_string())
}

fn fingers(planner: &mut Planner, opening: f64) -> Result<(), Failure> {
    planner
        .fingers(opening)
        .map_err(|error| Failure::Plan(error.to_string()))
}

/// Opens the hand, in the simulator and in the scene. A hand left low by a
/// missed grasp then backs off 10 cm straight up, the open fingers allowed to
/// brush the cube they missed.
pub fn open(clock: &Arc<ManualClock>) -> Act<(), ()> {
    let clock = clock.clone();
    act("open", move |arm, planner, (): ()| {
        grip(arm, false)?;
        fingers(planner, OPEN)?;
        let q = joints(arm)?;
        let tip = planner.tip(&q)?;
        if tip.translation.z > REST + ABOVE / 2.0 {
            return Ok(());
        }
        let up = Isometry3::translation(0.0, 0.0, 0.10) * tip;
        let goal = planner.reach_pose(up, q)?;
        straddle(planner, true);
        let moved = move_to(arm, planner, &clock, goal, SPEED);
        straddle(planner, false);
        moved.map(|_| ())
    })
}

/// Reads where the cube is, adding `errors[n]` (m, x and y) to the n-th
/// reading: a stand-in for perception that is wrong when told to be.
pub fn perceive(cube: i32, errors: &'static [[f64; 2]]) -> Act<(), Isometry3<f64>> {
    let readings = Arc::new(Mutex::new(0_usize));
    act("perceive", move |arm, _, (): ()| {
        let mut count = readings.lock().unwrap();
        let [dx, dy] = errors.get(*count).copied().unwrap_or([0.0, 0.0]);
        *count += 1;
        let mut pose = Cell::cube_pose(arm, cube)?;
        pose.translation.x += dx;
        pose.translation.y += dy;
        Ok(pose)
    })
}

/// Lets the open fingers touch the cube, as they straddle it, or forbids it.
fn straddle(planner: &mut Planner, allowed: bool) {
    for finger in FINGER_LINKS {
        if allowed {
            planner.scene.allowed_mut().allow("cube", finger);
        } else {
            planner.scene.allowed_mut().forbid("cube", finger);
        }
    }
}

/// The grasp configurations for holding the perceived cube the `yaw` way and
/// setting it down at `place` (x, y), checked in the scene with the fingers
/// open around the cube.
pub fn plan_grasp(place: [f64; 2]) -> Act<(Isometry3<f64>, Yaw), Grasp> {
    act(
        "plan the grasp",
        move |arm, planner, (cube, yaw): (Isometry3<f64>, Yaw)| {
            let [x, y] = [cube.translation.x, cube.translation.y];
            planner
                .scene
                .move_object("cube", Isometry3::translation(x, y, REST))
                .map_err(scene)?;
            let seed = joints(arm)?;
            let at = planner.reach([x, y, REST], yaw, seed)?;
            let above = planner.reach([x, y, REST + ABOVE], yaw, at)?;
            straddle(planner, true);
            let blocked = [("the grasp", &at), ("above it", &above)]
                .into_iter()
                .find_map(|(name, q)| Some((name, planner.scene.first_contact(q)?)));
            straddle(planner, false);
            if let Some((name, contact)) = blocked {
                return Err(Failure::Blocked(format!("{name}: {contact:?}")));
            }
            let at_place = planner.reach([place[0], place[1], REST], yaw, at)?;
            let above_place = planner.reach([place[0], place[1], REST + ABOVE], yaw, at_place)?;
            Ok(Grasp {
                above,
                at,
                above_place,
                at_place,
            })
        },
    )
}

/// A planned motion to the configuration `goal` picks out of the grasp.
fn motion(
    name: &'static str,
    clock: &Arc<ManualClock>,
    goal: fn(&Grasp) -> [f64; 7],
) -> Act<Grasp, Grasp> {
    let clock = clock.clone();
    act(name, move |arm, planner, grasp: Grasp| {
        move_to(arm, planner, &clock, goal(&grasp), SPEED)?;
        Ok(grasp)
    })
}

pub fn approach(clock: &Arc<ManualClock>) -> Act<Grasp, Grasp> {
    motion("approach", clock, |grasp| grasp.above)
}

/// Lowers the open fingers around the cube.
pub fn descend(clock: &Arc<ManualClock>) -> Act<Grasp, Grasp> {
    let clock = clock.clone();
    act("descend", move |arm, planner, grasp: Grasp| {
        straddle(planner, true);
        move_to(arm, planner, &clock, grasp.at, SPEED)?;
        Ok(grasp)
    })
}

/// Closes on the cube; if it is held, the scene carries it in the hand.
pub fn close(cube: i32) -> Act<Grasp, Grasp> {
    act("close", move |arm, planner, grasp: Grasp| {
        let width = grip(arm, true)?;
        planner
            .scene
            .move_object("cube", Cell::cube_pose(arm, cube)?)
            .map_err(scene)?;
        fingers(planner, width / 2.0)?;
        planner
            .scene
            .attach("cube", "panda_hand", &FINGER_LINKS, &grasp.at)
            .map_err(scene)?;
        Ok(grasp)
    })
}

/// Moves with the cube allowed to touch the table, as it leaves or meets it.
fn near_table(
    name: &'static str,
    clock: &Arc<ManualClock>,
    goal: fn(&Grasp) -> [f64; 7],
) -> Act<Grasp, Grasp> {
    let clock = clock.clone();
    act(name, move |arm, planner, grasp: Grasp| {
        planner.scene.allowed_mut().allow("cube", "table");
        let moved = move_to(arm, planner, &clock, goal(&grasp), SPEED);
        planner.scene.allowed_mut().forbid("cube", "table");
        moved?;
        still_held(arm)?;
        Ok(grasp)
    })
}

/// Fails if the cube is no longer between the fingers.
fn still_held(arm: &mut Device) -> Result<(), Failure> {
    let state = arm
        .read_gripper()
        .map_err(|error| Failure::Device(error.to_string()))?;
    if state.grasped == 1 {
        Ok(())
    } else {
        Err(Failure::Slipped { width: state.width })
    }
}

pub fn lift(clock: &Arc<ManualClock>) -> Act<Grasp, Grasp> {
    near_table("lift", clock, |grasp| grasp.above)
}

/// Carries the cube to above its place, over the wall.
pub fn carry(clock: &Arc<ManualClock>) -> Act<Grasp, Grasp> {
    let clock = clock.clone();
    act("carry over the wall", move |arm, planner, grasp: Grasp| {
        move_to(arm, planner, &clock, grasp.above_place, SPEED)?;
        still_held(arm)?;
        Ok(grasp)
    })
}

pub fn lower(clock: &Arc<ManualClock>) -> Act<Grasp, Grasp> {
    near_table("lower", clock, |grasp| grasp.at_place)
}

/// Opens the hand and leaves the cube in the scene where it was set down.
pub fn release() -> Act<Grasp, Grasp> {
    act("release", |arm, planner, grasp: Grasp| {
        grip(arm, false)?;
        planner
            .scene
            .detach("cube", &grasp.at_place)
            .map_err(scene)?;
        straddle(planner, false);
        fingers(planner, OPEN)?;
        Ok(grasp)
    })
}

pub fn retreat(clock: &Arc<ManualClock>) -> Act<Grasp, ()> {
    let clock = clock.clone();
    act("retreat", move |arm, planner, grasp: Grasp| {
        move_to(arm, planner, &clock, grasp.above_place, SPEED)?;
        Ok(())
    })
}

/// Where a trial left the cube.
#[derive(Clone, Copy, Debug)]
pub struct Placement {
    /// Distance (m) on the table from where the cube should be.
    pub off: f64,
    /// Whether the cube rests on the table.
    pub on_table: bool,
    /// Simulated time (s) since the trial started.
    pub seconds: f64,
}

/// Measures the cube against `place` (x, y), the trial having started at
/// simulated time `start` (s).
pub fn measure(cube: i32, place: [f64; 2]) -> Act<((), f64), Placement> {
    act("measure", move |arm, _, ((), start): ((), f64)| {
        let at = Cell::cube_pose(arm, cube)?.translation.vector;
        Ok(Placement {
            off: ((at.x - place[0]).powi(2) + (at.y - place[1]).powi(2)).sqrt(),
            on_table: (at.z - REST).abs() < 0.005,
            seconds: arm.engine().simulated_time().as_secs_f64() - start,
        })
    })
}

/// Sets up a trial: the hand opens and lets go in the scene, and the
/// simulator puts the cube at `at` (x, y) and the arm at `HOME`, as an
/// experimenter resets the cell. Returns the simulated time (s) the trial
/// starts at.
pub fn reset(cube: i32, body: i32, arm_joints: [i32; 7]) -> Act<[f64; 2], f64> {
    act("reset", move |arm, planner, at: [f64; 2]| {
        grip(arm, false)?;
        // A failed trial may have left the cube attached.
        let _ = planner.scene.detach("cube", &joints(arm)?);
        straddle(planner, false);
        planner.scene.allowed_mut().forbid("cube", "table");
        fingers(planner, OPEN)?;
        Cell::reset_arm(arm, body, arm_joints)?;
        Cell::place_cube(arm, cube, at)?;
        planner
            .scene
            .move_object("cube", Isometry3::translation(at[0], at[1], REST))
            .map_err(scene)?;
        Ok(arm.engine().simulated_time().as_secs_f64())
    })
}
