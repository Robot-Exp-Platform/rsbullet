//! A planned pick-and-place on the shared contracts.
//!
//! cargo run --release -p rsbullet --no-default-features --features roplat \
//!   --example panda_planned_pick -- /path/to/bullet3/examples/pybullet/gym/pybullet_data
//!
//! A Panda moves a cube over a wall on a table. Every arm motion is planned
//! with RRT-Connect (`plan_joint_path`) against a second Bullet world that
//! stands in for a planning scene, timed with COPP, and played by a
//! `JointTrajectoryFollower` in a `ControlRhythm` graph on an
//! `RsBulletDevice`; the hand is the device's `Gripper`. Only the planning
//! world is posed directly. The simulated world moves by motor commands
//! alone, and the run fails unless the cube ends within 5 mm of its place
//! and resting on the table. Build it with `--release`: in a debug build
//! COPP's solver takes about half a minute per motion.
//!
//! What the stand-in scene cannot do is the gap a real one fills: the cube
//! held in the hand is not part of it, so the transport keeps the arm a
//! margin from obstacles instead of checking the cube; near the table the
//! motions check the wall only, since there is no allowed-collision matrix;
//! and the arm's self-collision is not checked.
use std::{f64::consts::PI, sync::Arc, time::Duration};

use nalgebra::{Isometry3, Translation3, UnitQuaternion};
use roplat::{Completion, Execution, RoplatResult};
use roplat_behavior::behavior::{EndPoint, JointSpace, Joints};
use roplat_behavior::msgs::{GripperCommand, ManualClock};
use roplat_behavior::roplat::ControlRhythm;
use roplat_behavior::{
    AddRobot, EntityBuilder, Gripper, JointPositionControl, MoveTo, PhysicsEngine, RobotDescription,
};
use roplat_trajectory::{
    JointTrajectoryFollower, PlannerOptions, plan_joint_path, plan_waypoints_traj_via_copp,
    points_from_samples,
};
use rsbullet::{
    ClosestPointsOptions, CollisionGeometry, CollisionId, DynamicsUpdate, InverseKinematicsOptions,
    Mode, MultiBodyBase, MultiBodyCreateOptions, PhysicsClient, RsBullet, RsBulletDevice,
};

/// Limits from the Franka Emika Panda datasheet.
struct Panda;

impl RobotDescription for Panda {
    const URDF: Option<&'static str> = Some("franka_panda/panda.urdf");
}

impl Joints<7> for Panda {
    const JOINT_MIN: [f64; 7] = [
        -2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973,
    ];
    const JOINT_MAX: [f64; 7] = [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973];
    const JOINT_VEL_BOUND: [f64; 7] = [2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61];
    const JOINT_ACC_BOUND: [f64; 7] = [15.0, 7.5, 10.0, 12.5, 15.0, 20.0, 20.0];
    const JOINT_JERK_BOUND: [f64; 7] = [7500.0, 3750.0, 5000.0, 6250.0, 7500.0, 10000.0, 10000.0];
    const TORQUE_BOUND: [f64; 7] = [87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0];
}

impl EndPoint for Panda {}

type Device = RsBulletDevice<Panda>;
type Control = ControlRhythm<Device, JointPositionControl<7>>;
type Follower = JointTrajectoryFollower<Arc<ManualClock>, 7>;

const ARM: [&str; 7] = [
    "panda_joint1",
    "panda_joint2",
    "panda_joint3",
    "panda_joint4",
    "panda_joint5",
    "panda_joint6",
    "panda_joint7",
];
const FINGERS: [&str; 2] = ["panda_finger_joint1", "panda_finger_joint2"];
const HOME: [f64; 7] = [0.0, -0.6, 0.0, -2.1, 0.0, 1.6, 0.8];

const TABLE_TOP: f64 = 0.15;
const CUBE: f64 = 0.04;
const PICK: [f64; 2] = [0.45, -0.15];
const PLACE: [f64; 2] = [0.45, 0.20];
/// A 0.2 m high wall across the table between the two spots.
const WALL_HALF: [f64; 3] = [0.15, 0.01, 0.10];
const WALL_AT: [f64; 3] = [0.45, 0.03, TABLE_TOP + 0.10];
/// Height of the grasp point above the cube's centre before and after it.
const ABOVE: f64 = 0.15;
/// Kept between the arm and the obstacles while the cube, which the scene
/// does not know about, hangs from the hand.
const CARRY_MARGIN: f64 = 0.05;
/// Fraction of the joint velocity, acceleration and jerk limits.
const SPEED: f64 = 0.3;

/// A table and a wall, and the cube at its pick spot unless `cube` is unset.
#[derive(Clone, Copy)]
struct Furniture {
    table: i32,
    wall: i32,
    cube: Option<i32>,
}

impl Furniture {
    fn build(client: &mut PhysicsClient, cube: bool) -> anyhow::Result<Self> {
        let table = box_body(
            client,
            [0.35, 0.40, TABLE_TOP / 2.0],
            [0.55, 0.0, TABLE_TOP / 2.0],
            0.0,
        )?;
        let wall = box_body(client, WALL_HALF, WALL_AT, 0.0)?;
        let cube = if cube {
            let at = [PICK[0], PICK[1], TABLE_TOP + CUBE / 2.0];
            let cube = box_body(client, [CUBE / 2.0; 3], at, 0.05)?;
            for body in [table, cube] {
                client.change_dynamics(
                    body,
                    -1,
                    &DynamicsUpdate {
                        lateral_friction: Some(0.8),
                        restitution: Some(0.0),
                        ..Default::default()
                    },
                )?;
            }
            Some(cube)
        } else {
            None
        };
        Ok(Self { table, wall, cube })
    }
}

fn box_body(
    client: &mut PhysicsClient,
    half: [f64; 3],
    at: [f64; 3],
    mass: f64,
) -> anyhow::Result<i32> {
    let shape = client.create_collision_shape(
        &CollisionGeometry::Box { half_extents: half },
        None::<Isometry3<f64>>,
    )?;
    Ok(client.create_multi_body(&MultiBodyCreateOptions {
        base: MultiBodyBase {
            mass,
            collision_shape: CollisionId(shape),
            pose: Isometry3::translation(at[0], at[1], at[2]),
            ..Default::default()
        },
        ..Default::default()
    })?)
}

/// A second Bullet world holding the Panda and the furniture, posed directly
/// to answer inverse kinematics and clearance queries for the planner.
struct PlanningScene {
    world: RsBullet,
    panda: i32,
    arm: Vec<i32>,
    fingers: [i32; 2],
    hand: i32,
    furniture: Furniture,
}

impl PlanningScene {
    fn new(assets: &str) -> anyhow::Result<Self> {
        let mut world = RsBullet::new(Mode::Direct)?;
        world.client.set_additional_search_path(assets)?;
        let furniture = Furniture::build(&mut world.client, true)?;
        let robot = world
            .robot_builder::<Panda>("panda")
            .joint_names(ARM)
            .end_effector_link_name("panda_grasptarget")
            .base_fixed(true)
            .load()?;
        let mut fingers = [0; 2];
        for joint in 0..world.client.get_num_joints(robot.body_id) {
            let name = world
                .client
                .get_joint_info(robot.body_id, joint)?
                .joint_name;
            if let Some(k) = FINGERS.iter().position(|finger| *finger == name) {
                fingers[k] = joint;
            }
        }
        Ok(Self {
            panda: robot.body_id,
            arm: robot.joint_indices.clone(),
            hand: robot.end_effector_link,
            fingers,
            furniture,
            world,
        })
    }

    /// Poses the arm at `q` with each finger `finger` open.
    fn pose(&mut self, q: &[f64; 7], finger: f64) -> anyhow::Result<()> {
        let client = &mut self.world.client;
        for (&joint, &position) in self.arm.iter().zip(q) {
            client.reset_joint_state(self.panda, joint, position, None)?;
        }
        for joint in self.fingers {
            client.reset_joint_state(self.panda, joint, finger, None)?;
        }
        Ok(())
    }

    /// Whether the arm at `q` keeps `margin` from every body in `obstacles`.
    fn clear(&mut self, q: &[f64; 7], finger: f64, margin: f64, obstacles: &[i32]) -> bool {
        self.pose(q, finger).is_ok()
            && obstacles.iter().all(|&obstacle| {
                self.world
                    .client
                    .get_closest_points(
                        self.panda,
                        obstacle,
                        margin,
                        &ClosestPointsOptions::default(),
                    )
                    .is_ok_and(|points| points.is_empty())
            })
    }

    /// Joint positions placing the grasp point at `at`, pointing down,
    /// starting the solver from `seed`.
    fn reach(&mut self, at: [f64; 3], seed: [f64; 7]) -> anyhow::Result<[f64; 7]> {
        self.pose(&seed, 0.04)?;
        let mut current: Vec<f64> = seed.to_vec();
        current.extend([0.04; 2]);
        let target = Isometry3::from_parts(
            Translation3::from(at),
            UnitQuaternion::from_euler_angles(PI, 0.0, 0.0),
        );
        let solution = self.world.client.calculate_inverse_kinematics(
            self.panda,
            self.hand,
            target,
            &InverseKinematicsOptions {
                current_positions: Some(&current),
                max_iterations: Some(200),
                residual_threshold: Some(1e-6),
                ..Default::default()
            },
        )?;
        anyhow::ensure!(solution.len() >= 7, "IK returned {} joints", solution.len());
        let q: [f64; 7] = std::array::from_fn(|i| solution[i]);
        anyhow::ensure!(
            (0..7).all(|i| (Panda::JOINT_MIN[i]..=Panda::JOINT_MAX[i]).contains(&q[i])),
            "IK for {at:?} left the joint limits: {q:?}"
        );
        self.pose(&q, 0.04)?;
        let reached = self
            .world
            .client
            .get_link_state(self.panda, self.hand, false, true)?
            .world_link_frame
            .translation
            .vector;
        let miss = (reached - nalgebra::Vector3::from(at)).norm();
        anyhow::ensure!(miss < 0.005, "IK for {at:?} missed by {miss:.4} m");
        Ok(q)
    }

    /// A collision-free path from `start` to `goal` with the fingers at
    /// `finger`, keeping `margin` from `obstacles`.
    fn plan(
        &mut self,
        start: [f64; 7],
        goal: [f64; 7],
        finger: f64,
        margin: f64,
        obstacles: &[i32],
    ) -> anyhow::Result<Vec<[f64; 7]>> {
        let path = plan_joint_path(
            start,
            goal,
            Panda::JOINT_MIN,
            Panda::JOINT_MAX,
            |q| self.clear(q, finger, margin, obstacles),
            PlannerOptions {
                max_iterations: 20_000,
                seed: 20261028,
                ..Default::default()
            },
        )?;
        Ok(path)
    }
}

/// Plays one trajectory on the arm's position channel; the follower refuses
/// with a tracking error if the arm leaves the path.
#[roplat::system]
async fn follow(
    mut control: Control,
    mut follower: Follower,
    mut arm: Device,
) -> RoplatResult<(Execution<()>, Device, Control)> {
    &mut arm >> control.resources.0;
    control
        >> |sample| {
            sample >> follower;
            follower?
        };
    Ok((control.outcome, arm, control))
}

/// Times `path` with COPP and follows it to the end.
async fn run(
    name: &str,
    path: &[[f64; 7]],
    arm: Device,
    control: Control,
    clock: &Arc<ManualClock>,
) -> anyhow::Result<(Device, Control)> {
    let samples = plan_waypoints_traj_via_copp::<Device, 7>(path, SPEED)?;
    let period = Duration::from_secs_f64(<Device as roplat_behavior::Robot>::CONTROL_PERIOD);
    let follower =
        JointTrajectoryFollower::new(points_from_samples(&samples, period), clock.clone())?
            .with_path_tolerance([0.1; 7])
            .with_goal_tolerance([0.01; 7], Duration::from_secs(1));
    let duration = follower.duration();
    let (outcome, arm, control) = follow(control, follower, arm).await?;
    anyhow::ensure!(
        matches!(outcome, Ok(Completion::Completed(()))),
        "{name}: {outcome:?}"
    );
    println!(
        "{name:<9} {:>2} waypoints, {:>5.2} s planned, sim time {:>6.2} s",
        path.len(),
        duration.as_secs_f64(),
        arm.engine().simulated_time().as_secs_f64(),
    );
    Ok((arm, control))
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let assets = args
        .next()
        .ok_or_else(|| anyhow::anyhow!("supply the pinned Bullet model directory"))?;
    anyhow::ensure!(args.next().is_none(), "expected only MODEL_DIRECTORY");

    let mut world = RsBullet::new(Mode::Direct)?;
    world.set_gravity([0.0, 0.0, -9.81])?;
    world.client.set_additional_search_path(&assets)?;
    let furniture = Furniture::build(&mut world.client, true)?;
    let cube = furniture.cube.expect("built with the cube");
    let robot = world
        .robot_builder::<Panda>("panda")
        .joint_names(ARM)
        .end_effector_link_name("panda_grasptarget")
        .base_fixed(true)
        .load()?;
    for (&joint, &q) in robot.joint_indices.iter().zip(&HOME) {
        world
            .client
            .reset_joint_state(robot.body_id, joint, q, None)?;
    }
    let clock = world.clock();
    let mut arm = RsBulletDevice::new(world, robot)?.with_gripper(FINGERS)?;
    MoveTo::<JointSpace<7>>::move_to(&mut arm, HOME)?;
    arm.home_gripper()?;

    let mut scene = PlanningScene::new(&assets)?;
    let Furniture {
        table,
        wall,
        cube: Some(scene_cube),
    } = scene.furniture
    else {
        unreachable!("the planning scene is built with the cube")
    };
    let rest = TABLE_TOP + CUBE / 2.0;
    let at_pick = scene.reach([PICK[0], PICK[1], rest], HOME)?;
    let above_pick = scene.reach([PICK[0], PICK[1], rest + ABOVE], at_pick)?;
    let at_place = scene.reach([PLACE[0], PLACE[1], rest], at_pick)?;
    let above_place = scene.reach([PLACE[0], PLACE[1], rest + ABOVE], at_place)?;
    let (open, held) = (0.04, CUBE / 2.0);

    let mut control = Control::new();
    let path = scene.plan(HOME, above_pick, open, 0.01, &[table, wall, scene_cube])?;
    (arm, control) = run("approach", &path, arm, control, &clock).await?;
    let path = scene.plan(above_pick, at_pick, open, 0.01, &[wall])?;
    (arm, control) = run("descend", &path, arm, control, &clock).await?;

    arm.move_gripper(GripperCommand {
        width: CUBE,
        speed: 0.05,
        force: 15.0,
    })?;
    let grip = arm.read_gripper()?;
    anyhow::ensure!(grip.grasped == 1, "the grasp failed: {grip:?}");
    println!("grasped   width {:.4} m", grip.width);

    let path = scene.plan(at_pick, above_pick, held, 0.01, &[wall])?;
    (arm, control) = run("lift", &path, arm, control, &clock).await?;
    let path = scene.plan(above_pick, above_place, held, CARRY_MARGIN, &[table, wall])?;
    (arm, control) = run("transport", &path, arm, control, &clock).await?;
    anyhow::ensure!(
        arm.read_gripper()?.grasped == 1,
        "the cube slipped in transport"
    );
    let path = scene.plan(above_place, at_place, held, 0.01, &[wall])?;
    (arm, control) = run("lower", &path, arm, control, &clock).await?;

    arm.move_gripper(GripperCommand {
        width: 0.08,
        speed: 0.05,
        force: 0.0,
    })?;
    let path = scene.plan(at_place, above_place, open, 0.01, &[wall])?;
    (arm, _) = run("retreat", &path, arm, control, &clock).await?;
    for _ in 0..240 {
        arm.engine_mut().step()?;
    }

    let placed = arm
        .engine_mut()
        .client
        .get_base_position_and_orientation(cube)?
        .translation
        .vector;
    let off = ((placed.x - PLACE[0]).powi(2) + (placed.y - PLACE[1]).powi(2)).sqrt();
    println!(
        "cube at ({:.3}, {:.3}, {:.3}), {:.1} mm from its place, after {} integrations",
        placed.x,
        placed.y,
        placed.z,
        off * 1e3,
        arm.engine().steps_completed(),
    );
    anyhow::ensure!(off < 0.005, "the cube ended {off:.3} m from its place");
    anyhow::ensure!(
        (placed.z - rest).abs() < 0.005,
        "the cube is not on the table"
    );
    Ok(())
}
