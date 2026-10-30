//! A hand on the device's body behind the shared `Gripper` contract: its
//! motions step the world until the fingers rest, a grasp holds a block
//! through a lift, a grasp that closes on nothing is reported rather than
//! raised, and a blocked move is an error.

use std::sync::{Arc, Mutex};

use nalgebra::Isometry3;
use roplat_behavior::behavior::{EndPoint, JointSpace, Joints};
use roplat_behavior::msgs::GripperCommand;
use roplat_behavior::{Gripper, MoveTo, RobotDescription, RobotException};
use roplat_world::{AddRobot, EntityBuilder, PhysicsEngine};
use rsbullet::{ForceFrame, GRASP_EPSILON, MOVE_TOLERANCE, Mode, RsBullet, RsBulletDevice};

/// A vertical lift carrying a hand of two 0.04 m fingers (effort 20 N).
struct Rig;

impl RobotDescription for Rig {
    const URDF: Option<&'static str> = Some(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/assets/gripper_rig.urdf"
    ));
}

impl Joints<1> for Rig {
    const JOINT_MIN: [f64; 1] = [-0.2];
    const JOINT_MAX: [f64; 1] = [0.2];
    const JOINT_VEL_BOUND: [f64; 1] = [0.5];
    const JOINT_ACC_BOUND: [f64; 1] = [2.0];
    const JOINT_JERK_BOUND: [f64; 1] = [20.0];
}

impl EndPoint for Rig {}

/// A 0.04 m, 0.1 kg cube.
struct Block;

impl RobotDescription for Block {
    const URDF: Option<&'static str> = Some(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/assets/block.urdf"
    ));
}

const FINGERS: [&str; 2] = ["finger_left", "finger_right"];
/// Height of the block's centre resting on the table.
const ON_TABLE: f64 = 0.26;

fn command(width: f64, force: f64) -> GripperCommand {
    GripperCommand { width, speed: 0.1, force }
}

/// The rig under gravity with its fingers above a table, before any hand.
fn rig() -> anyhow::Result<RsBulletDevice<Rig>> {
    let mut engine = RsBullet::new(Mode::Direct)?;
    engine.set_gravity([0.0, 0.0, -9.81])?;
    let rig = engine
        .robot_builder::<Rig>("rig")
        .joint_names(["lift"])
        .base_fixed(true)
        .load()?;
    // The block scaled to a 0.2 m table whose top is at 0.24 m.
    engine
        .robot_builder::<Block>("table")
        .base(Isometry3::translation(0.0, 0.0, 0.14))
        .scaling(5.0)
        .base_fixed(true)
        .load()?;
    Ok(RsBulletDevice::new(engine, rig)?)
}

/// The rig holding its lift with the hand open.
fn scene() -> anyhow::Result<RsBulletDevice<Rig>> {
    let mut device = rig()?.with_gripper(FINGERS)?;
    MoveTo::<JointSpace<1>>::move_to(&mut device, [0.0])?;
    device.home_gripper()?;
    Ok(device)
}

/// The block and its position, sampled after every step.
struct Tracked {
    body: i32,
    at: Arc<Mutex<[f64; 3]>>,
}

impl Tracked {
    fn at(&self) -> [f64; 3] {
        *self.at.lock().unwrap()
    }
}

/// [`scene`] with the block resting on the table between the fingers.
fn scene_with_block() -> anyhow::Result<(RsBulletDevice<Rig>, Tracked)> {
    let mut device = scene()?;
    let engine = device.engine_mut();
    let body = engine
        .robot_builder::<Block>("block")
        .base(Isometry3::translation(0.0, 0.0, ON_TABLE))
        .load()?
        .body_id;
    let at = Arc::new(Mutex::new([f64::NAN; 3]));
    let sink = at.clone();
    engine.add_step_observer(move |client| {
        let position = client.get_base_position_and_orientation(body)?.translation;
        *sink.lock().unwrap() = [position.x, position.y, position.z];
        Ok(())
    });
    for _ in 0..120 {
        engine.step()?;
    }
    Ok((device, Tracked { body, at }))
}

fn unprocessable<T>(result: Result<T, RobotException>) -> bool {
    matches!(
        result,
        Err(RobotException::UnprocessableInstructionError(_))
    )
}

#[test]
fn attaching_checks_the_finger_joints() -> anyhow::Result<()> {
    assert!(unprocessable(
        rig()?.with_gripper(["finger_left", "finger_left"])
    ));
    assert!(unprocessable(rig()?.with_gripper(["finger_left", "lift"])));
    assert!(unprocessable(rig()?.with_gripper(["finger_left", "thumb"])));
    let twice = rig()?.with_gripper(FINGERS)?.with_gripper(FINGERS);
    assert!(unprocessable(twice));
    let mut bare = rig()?;
    assert!(unprocessable(bare.read_gripper()));
    assert!(unprocessable(bare.move_gripper(command(0.02, 0.0))));
    Ok(())
}

#[test]
fn homing_opens_and_a_free_move_reaches_its_width() -> anyhow::Result<()> {
    let mut device = scene()?;
    let open = device.read_gripper()?;
    assert!((open.width - 0.08).abs() <= MOVE_TOLERANCE, "{open:?}");
    // Bullet keeps joint limits in single precision.
    assert!((open.max_width - 0.08).abs() < 1e-6, "{open:?}");
    assert_eq!(open.grasped, 0);

    let before = device.engine().steps_completed();
    device.move_gripper(command(0.03, 0.0))?;
    // 0.05 m at 0.1 m/s takes half a second of simulated time.
    assert!(device.engine().steps_completed() - before >= 120);
    let moved = device.read_gripper()?;
    assert!((moved.width - 0.03).abs() <= MOVE_TOLERANCE, "{moved:?}");
    assert_eq!(moved.grasped, 0);
    Ok(())
}

#[test]
fn a_grasp_holds_the_block_through_a_lift_and_a_release_drops_it() -> anyhow::Result<()> {
    let (mut device, block) = scene_with_block()?;
    device.move_gripper(command(0.04, 10.0))?;
    let held = device.read_gripper()?;
    assert_eq!(held.grasped, 1, "{held:?}");
    assert!((held.width - 0.04).abs() <= GRASP_EPSILON, "{held:?}");

    MoveTo::<JointSpace<1>>::move_to(&mut device, [0.1])?;
    let lifted = block.at()[2];
    assert!(
        (lifted - (ON_TABLE + 0.1)).abs() < 0.01,
        "block at {lifted}"
    );
    assert_eq!(device.read_gripper()?.grasped, 1);

    device.move_gripper(command(0.08, 0.0))?;
    assert_eq!(device.read_gripper()?.grasped, 0);
    for _ in 0..240 {
        device.engine_mut().step()?;
    }
    let dropped = block.at()[2];
    assert!(dropped < ON_TABLE + 0.02, "block at {dropped}");
    Ok(())
}

/// Two force-limited fingers alone would let a squeezed block slide along
/// the closing axis; the coupling holds them mirrored.
#[test]
fn the_coupled_fingers_hold_a_block_against_a_sideways_push() -> anyhow::Result<()> {
    let (mut device, block) = scene_with_block()?;
    device.move_gripper(command(0.04, 10.0))?;
    MoveTo::<JointSpace<1>>::move_to(&mut device, [0.05])?;
    let before = block.at();
    let (body, mut pushes) = (block.body, 240);
    // An observer's force acts during the next step: 3 N along +y for 1 s.
    device.engine_mut().add_step_observer(move |client| {
        if pushes > 0 {
            pushes -= 1;
            let at = client.get_base_position_and_orientation(body)?.translation;
            client.apply_external_force(
                body,
                -1,
                [0.0, 3.0, 0.0],
                [at.x, at.y, at.z],
                ForceFrame::World,
            )?;
        }
        Ok(())
    });
    for _ in 0..240 {
        device.engine_mut().step()?;
    }
    let slid = block.at()[1] - before[1];
    assert!(slid.abs() < 2e-3, "the block slid {slid} m");
    assert_eq!(device.read_gripper()?.grasped, 1);
    Ok(())
}

#[test]
fn a_grasp_that_closes_on_nothing_is_reported_not_raised() -> anyhow::Result<()> {
    let mut device = scene()?;
    device.move_gripper(command(0.04, 10.0))?;
    let state = device.read_gripper()?;
    assert_eq!(state.grasped, 0, "{state:?}");
    assert!(state.width < 0.04 - GRASP_EPSILON, "{state:?}");
    Ok(())
}

#[test]
fn a_move_blocked_by_the_block_is_an_error() -> anyhow::Result<()> {
    let (mut device, _) = scene_with_block()?;
    let blocked = device.move_gripper(command(0.01, 0.0));
    assert!(
        matches!(blocked, Err(RobotException::CommandException(_))),
        "{blocked:?}"
    );
    let state = device.read_gripper()?;
    assert!((state.width - 0.04).abs() < 0.005, "{state:?}");
    assert_eq!(state.grasped, 0);
    Ok(())
}

#[test]
fn stopping_releases_a_grasp() -> anyhow::Result<()> {
    let (mut device, _) = scene_with_block()?;
    device.move_gripper(command(0.04, 10.0))?;
    assert_eq!(device.read_gripper()?.grasped, 1);
    let before = device.engine().steps_completed();
    device.stop_gripper()?;
    assert_eq!(device.engine().steps_completed(), before);
    assert_eq!(device.read_gripper()?.grasped, 0);
    Ok(())
}

#[test]
fn commands_outside_the_hand_are_refused_without_stepping() -> anyhow::Result<()> {
    let mut device = scene()?;
    let before = device.engine().steps_completed();
    for refused in [
        command(0.09, 0.0),
        command(-0.01, 0.0),
        command(f64::NAN, 0.0),
        GripperCommand { width: 0.04, speed: 0.0, force: 0.0 },
        GripperCommand { width: 0.04, speed: f64::NAN, force: 0.0 },
        command(0.04, 25.0),
        command(0.04, -1.0),
    ] {
        assert!(unprocessable(device.move_gripper(refused)), "{refused:?}");
    }
    assert_eq!(device.engine().steps_completed(), before);
    Ok(())
}
