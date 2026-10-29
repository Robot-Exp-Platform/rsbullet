//! A planned pick-and-place on the shared contracts.
//!
//! cargo run --release -p rsbullet --no-default-features --features roplat \
//!   --example panda_planned_pick -- /path/to/bullet3/examples/pybullet/gym/pybullet_data
//!
//! A Panda moves a cube over a wall on a table. Planning does not use
//! Bullet: `roplat_description` reads the same URDF, `chain_ik` from
//! `roplat_kinematics` finds the grasp configurations, and every arm motion
//! is planned with RRT-Connect (`plan_joint_path`) against a
//! `roplat_scene::PlanningScene` of the table, the wall and the cube, which
//! also checks the arm against itself and keeps a 1 cm padding. Each path is
//! timed with COPP and played by a `JointTrajectoryFollower` in a
//! `ControlRhythm` graph on an `RsBulletDevice`; the hand is the device's
//! `Gripper`. The simulated world moves by motor commands alone, and the run
//! fails unless the cube ends within 5 mm of its place and resting on the
//! table. Build it with `--release`: in a debug build COPP's solver takes
//! about half a minute per motion.
//!
//! The scene follows the task. The open fingers may touch the cube on the way
//! down. After the grasp the scene takes the cube's pose from the simulator,
//! standing in for perception, and attaches the cube to the hand, so the
//! transport checks the cube itself against the wall and the table. The cube
//! may touch the table only while it is lifted off and set down.
use std::{f64::consts::PI, path::Path, sync::Arc, time::Duration};

use nalgebra::{Isometry3, Translation3, UnitQuaternion};
use roplat::{Completion, Execution, RoplatResult};
use roplat_behavior::behavior::{EndPoint, JointSpace, Joints};
use roplat_behavior::msgs::{GripperCommand, ManualClock};
use roplat_behavior::roplat::ControlRhythm;
use roplat_behavior::{Gripper, JointPositionControl, MoveTo, RobotDescription};
use roplat_description::{Chain, Geometry, RobotModel};
use roplat_kinematics::{CommonStop, chain_ik};
use roplat_scene::{CollisionCheck, PlanningScene};
use roplat_trajectory::{
    JointTrajectoryFollower, PlannerOptions, plan_joint_path, plan_waypoints_traj_via_copp,
    points_from_samples,
};
use roplat_world::{AddRobot, EntityBuilder, PhysicsEngine};
use rsbullet::{
    CollisionGeometry, CollisionId, DynamicsUpdate, Mode, MultiBodyBase, MultiBodyCreateOptions,
    PhysicsClient, RsBullet, RsBulletDevice,
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

impl EndPoint for Panda {
    // A model description; the simulated robot reports its pose.
    fn get_endpoint(&self) -> roplat_behavior::Pose {
        unreachable!("Panda describes a model and is never a robot")
    }
}

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
const FINGER_LINKS: [&str; 2] = ["panda_leftfinger", "panda_rightfinger"];
const HOME: [f64; 7] = [0.0, -0.6, 0.0, -2.1, 0.0, 1.6, 0.8];

const TABLE_TOP: f64 = 0.15;
const TABLE_HALF: [f64; 3] = [0.35, 0.40, TABLE_TOP / 2.0];
const TABLE_AT: [f64; 3] = [0.55, 0.0, TABLE_TOP / 2.0];
const CUBE: f64 = 0.04;
const PICK: [f64; 2] = [0.45, -0.15];
const PLACE: [f64; 2] = [0.45, 0.20];
/// A 0.2 m high wall across the table between the two spots.
const WALL_HALF: [f64; 3] = [0.15, 0.01, 0.10];
const WALL_AT: [f64; 3] = [0.45, 0.03, TABLE_TOP + 0.10];
/// Height of the grasp point above the cube's centre before and after it.
const ABOVE: f64 = 0.15;
/// Clearance (m) the plans keep between bodies that may not touch.
const PADDING: f64 = 0.01;
/// Fraction of the joint velocity, acceleration and jerk limits.
const SPEED: f64 = 0.3;

/// The table, the wall and the cube at its pick spot; returns the cube.
fn furnish(client: &mut PhysicsClient) -> anyhow::Result<i32> {
    let table = box_body(client, TABLE_HALF, TABLE_AT, 0.0)?;
    box_body(client, WALL_HALF, WALL_AT, 0.0)?;
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
    Ok(cube)
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

/// The planning side, from the same URDF: a scene of the furniture and the
/// chain to the grasp point.
struct Planner {
    scene: PlanningScene<7>,
    grasp_point: Chain,
}

impl Planner {
    fn new(assets: &Path) -> anyhow::Result<Self> {
        let dir = assets.join("franka_panda");
        let model = RobotModel::from_urdf_file(dir.join("panda.urdf"))?;
        let grasp_point = model.chain("panda_link0", "panda_grasptarget")?;
        // Bullet's data names meshes `package://meshes/…` beside the URDF.
        let mut scene = PlanningScene::new(model, ARM, |name| {
            dir.join(name.trim_start_matches("package://"))
        })?;
        let block = |half: [f64; 3]| Geometry::Box {
            size: half.map(|h| 2.0 * h),
        };
        let at = |[x, y, z]: [f64; 3]| Isometry3::translation(x, y, z);
        scene.add_object("table", &block(TABLE_HALF), at(TABLE_AT))?;
        scene.add_object("wall", &block(WALL_HALF), at(WALL_AT))?;
        let rest = [PICK[0], PICK[1], TABLE_TOP + CUBE / 2.0];
        scene.add_object("cube", &block([CUBE / 2.0; 3]), at(rest))?;
        scene.set_padding(PADDING)?;
        Ok(Self { scene, grasp_point })
    }

    /// Sets each finger's opening (m) in the scene.
    fn fingers(&mut self, opening: f64) -> anyhow::Result<()> {
        for finger in FINGERS {
            self.scene.set_joint(finger, opening)?;
        }
        Ok(())
    }

    /// Joint positions placing the grasp point at `at`, pointing down,
    /// starting the solver from `seed`.
    fn reach(&self, at: [f64; 3], seed: [f64; 7]) -> anyhow::Result<[f64; 7]> {
        let target = Isometry3::from_parts(
            Translation3::from(at),
            UnitQuaternion::from_euler_angles(PI, 0.0, 0.0),
        );
        let stop = CommonStop {
            pos_tol: 1e-5,
            rot_tol: 1e-4,
            max_iters: 1000,
            step_clip: 0.2,
        };
        let solution = chain_ik(&self.grasp_point, &seed, &target.into(), 1e-3, stop)?;
        anyhow::ensure!(
            solution.converged,
            "IK for {at:?} stopped {:.4} m and {:.4} rad off",
            solution.position_error,
            solution.rotation_error
        );
        let q: [f64; 7] = std::array::from_fn(|i| solution.q[i]);
        anyhow::ensure!(
            (0..7).all(|i| (Panda::JOINT_MIN[i]..=Panda::JOINT_MAX[i]).contains(&q[i])),
            "IK for {at:?} left the joint limits: {q:?}"
        );
        Ok(q)
    }

    /// A path from `start` to `goal` that is free in the scene as it is now.
    fn plan(&self, start: [f64; 7], goal: [f64; 7]) -> anyhow::Result<Vec<[f64; 7]>> {
        let options = PlannerOptions {
            max_iterations: 20_000,
            seed: 20261028,
            ..Default::default()
        };
        plan_joint_path(
            start,
            goal,
            Panda::JOINT_MIN,
            Panda::JOINT_MAX,
            |q| self.scene.is_free(q),
            options,
        )
        .map_err(|error| {
            anyhow::anyhow!(
                "{error}; at the start {:?}, at the goal {:?}",
                self.scene.first_contact(&start),
                self.scene.first_contact(&goal)
            )
        })
    }

    /// The smallest distance between bodies that may not touch, over the
    /// waypoints of `path`.
    fn clearance(&self, path: &[[f64; 7]]) -> f64 {
        path.iter()
            .filter_map(|q| self.scene.clearance(q))
            .map(|nearest| nearest.distance)
            .fold(f64::INFINITY, f64::min)
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

/// The simulated arm and its control rhythm, moved one planned motion at a
/// time.
struct Rig {
    arm: Device,
    control: Control,
    clock: Arc<ManualClock>,
}

impl Rig {
    /// Plans from `start` to `goal` in the scene, times the path with COPP
    /// and follows it to the end.
    async fn run(
        self,
        name: &str,
        planner: &Planner,
        [start, goal]: [[f64; 7]; 2],
    ) -> anyhow::Result<Self> {
        let Self {
            arm,
            control,
            clock,
        } = self;
        let path = planner.plan(start, goal)?;
        let period = <Device as roplat_behavior::Robot>::CONTROL_PERIOD;
        let limits = roplat_behavior::MotionLimits::<7>::envelope::<Panda>()
            .scaled(SPEED)
            .scaled_joint();
        let samples = plan_waypoints_traj_via_copp(&path, &limits, period)?;
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
            "{name:<9} {:>2} waypoints, clearance {:>4.1} mm, {:>5.2} s planned, sim time {:>6.2} s",
            path.len(),
            planner.clearance(&path) * 1e3,
            duration.as_secs_f64(),
            arm.engine().simulated_time().as_secs_f64(),
        );
        Ok(Self {
            arm,
            control,
            clock,
        })
    }
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
    let cube = furnish(&mut world.client)?;
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

    let mut planner = Planner::new(Path::new(&assets))?;
    let rest = TABLE_TOP + CUBE / 2.0;
    let at_pick = planner.reach([PICK[0], PICK[1], rest], HOME)?;
    let above_pick = planner.reach([PICK[0], PICK[1], rest + ABOVE], at_pick)?;
    let at_place = planner.reach([PLACE[0], PLACE[1], rest], at_pick)?;
    let above_place = planner.reach([PLACE[0], PLACE[1], rest + ABOVE], at_place)?;
    let open = 0.04;
    planner.fingers(open)?;

    let mut rig = Rig {
        arm,
        control: Control::new(),
        clock,
    };
    rig = rig.run("approach", &planner, [HOME, above_pick]).await?;
    // The open fingers straddle the cube on the way down.
    for finger in FINGER_LINKS {
        planner.scene.allowed_mut().allow("cube", finger);
    }
    rig = rig.run("descend", &planner, [above_pick, at_pick]).await?;

    rig.arm.move_gripper(GripperCommand {
        width: CUBE,
        speed: 0.05,
        force: 15.0,
    })?;
    let grip = rig.arm.read_gripper()?;
    anyhow::ensure!(grip.grasped == 1, "the grasp failed: {grip:?}");
    println!("grasped   width {:.4} m", grip.width);
    // The scene takes the cube where the simulator has it and carries it in
    // the hand.
    let held = rig
        .arm
        .engine_mut()
        .client
        .get_base_position_and_orientation(cube)?;
    planner.scene.move_object("cube", held)?;
    planner.fingers(grip.width / 2.0)?;
    planner
        .scene
        .attach("cube", "panda_hand", &FINGER_LINKS, &at_pick)?;

    // The cube leaves the table, is carried clear of it, and is set down.
    planner.scene.allowed_mut().allow("cube", "table");
    rig = rig.run("lift", &planner, [at_pick, above_pick]).await?;
    planner.scene.allowed_mut().forbid("cube", "table");
    rig = rig
        .run("transport", &planner, [above_pick, above_place])
        .await?;
    anyhow::ensure!(
        rig.arm.read_gripper()?.grasped == 1,
        "the cube slipped in transport"
    );
    planner.scene.allowed_mut().allow("cube", "table");
    rig = rig.run("lower", &planner, [above_place, at_place]).await?;

    rig.arm.move_gripper(GripperCommand {
        width: 0.08,
        speed: 0.05,
        force: 0.0,
    })?;
    planner.scene.detach("cube", &at_place)?;
    planner.fingers(open)?;
    let Rig { mut arm, .. } = rig
        .run("retreat", &planner, [at_place, above_place])
        .await?;
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
