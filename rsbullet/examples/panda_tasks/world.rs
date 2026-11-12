//! The cell shared by the tasks: a Panda on a table with a wall, a cube, and
//! the planning side built from the same URDF.
//!
//! Planning does not use Bullet. `Planner` reads the URDF with
//! `roplat_description`, solves grasp configurations with `chain_ik`, plans
//! with RRT-Connect against a `roplat_scene::PlanningScene`, and every motion
//! is timed with COPP and played by a `JointTrajectoryFollower` on the arm's
//! joint position channel. The simulator only executes.
use std::{
    f64::consts::{FRAC_PI_2, PI},
    ops::ControlFlow,
    path::Path,
    sync::Arc,
    time::Duration,
};

use nalgebra::{Isometry3, Translation3, UnitQuaternion};
use roplat_behavior::behavior::{EndPoint, JointSpace, Joints};
use roplat_behavior::msgs::{GripperCommand, ManualClock};
use roplat_behavior::{
    AddRobot, Arm, ControlWith, EntityBuilder, Gripper, JointPositionControl, MoveTo,
    PhysicsEngine, Robot, RobotDescription,
};
use roplat_description::{Chain, Geometry, RobotModel};
use roplat_kinematics::{CommonStop, chain_ik};
use roplat_scene::{CollisionCheck, PlanningScene};
use roplat_trajectory::{
    JointTrajectoryFollower, PlannerOptions, plan_joint_path, plan_waypoints_traj_via_copp,
    points_from_samples,
};
use rsbullet::{
    CollisionGeometry, CollisionId, DynamicsUpdate, Mode, MultiBodyBase, MultiBodyCreateOptions,
    PhysicsClient, RsBullet, RsBulletDevice,
};

use crate::act::Failure;

/// Limits from the Franka Emika Panda datasheet.
pub struct Panda;

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

pub type Device = RsBulletDevice<Panda>;
pub type Follower = JointTrajectoryFollower<Arc<ManualClock>, 7>;

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
pub const FINGER_LINKS: [&str; 2] = ["panda_leftfinger", "panda_rightfinger"];
pub const HOME: [f64; 7] = [0.0, -0.6, 0.0, -2.1, 0.0, 1.6, 0.8];

pub const TABLE_TOP: f64 = 0.15;
const TABLE_HALF: [f64; 3] = [0.35, 0.40, TABLE_TOP / 2.0];
const TABLE_AT: [f64; 3] = [0.55, 0.0, TABLE_TOP / 2.0];
pub const CUBE: f64 = 0.04;
/// Height of the cube's centre resting on the table.
pub const REST: f64 = TABLE_TOP + CUBE / 2.0;
/// A 0.2 m high wall across the table, its faces at y = 0.02 and 0.04.
const WALL_HALF: [f64; 3] = [0.15, 0.01, 0.10];
const WALL_AT: [f64; 3] = [0.45, 0.03, TABLE_TOP + 0.10];
/// Height of the grasp point above the cube's centre before and after it.
pub const ABOVE: f64 = 0.15;
/// Clearance (m) the plans keep between bodies that may not touch.
const PADDING: f64 = 0.01;
/// Fraction of the joint velocity, acceleration and jerk limits.
pub const SPEED: f64 = 0.3;
/// Finger opening (m, each) before a grasp.
pub const OPEN: f64 = 0.04;

/// How the hand meets the cube: the fingers close along world y at yaw 0 and
/// along world x at a quarter turn.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Yaw(pub f64);

impl Yaw {
    pub const ALONG_Y: Self = Self(0.0);
    pub const ALONG_X: Self = Self(FRAC_PI_2);
}

/// A fixed box on the table, besides the table and the wall.
#[derive(Clone, Copy, Debug)]
pub struct Block {
    pub name: &'static str,
    pub half: [f64; 3],
    pub at: [f64; 3],
}

/// The simulated cell: the world with the arm and the cube.
pub struct Cell {
    pub arm: Device,
    pub cube: i32,
    pub clock: Arc<ManualClock>,
    /// The arm's body and joint indices in the simulator, for resets.
    pub body: i32,
    pub arm_joints: [i32; 7],
}

impl Cell {
    /// Loads the table, the wall, the `blocks`, a cube resting at `at` (x, y)
    /// and the arm at `HOME` with the hand open.
    pub fn new(assets: &Path, at: [f64; 2], blocks: &[Block]) -> anyhow::Result<Self> {
        let mut world = RsBullet::new(Mode::Direct)?;
        world.set_gravity([0.0, 0.0, -9.81])?;
        world.client.set_additional_search_path(assets)?;
        let cube = furnish(&mut world.client, at)?;
        for block in blocks {
            box_body(&mut world.client, block.half, block.at, 0.0)?;
        }
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
        let body = robot.body_id;
        let arm_joints: [i32; 7] = std::array::from_fn(|i| robot.joint_indices[i]);
        let mut arm = RsBulletDevice::new(world, robot)?.with_gripper(FINGERS)?;
        MoveTo::<JointSpace<7>>::move_to(&mut arm, HOME)?;
        arm.home_gripper()?;
        Ok(Self {
            arm,
            cube,
            clock,
            body,
            arm_joints,
        })
    }

    /// Puts the arm at `HOME` and holds it there: the simulator moves it, not
    /// a motion.
    pub fn reset_arm(arm: &mut Device, body: i32, arm_joints: [i32; 7]) -> Result<(), Failure> {
        for (joint, q) in arm_joints.into_iter().zip(HOME) {
            arm.engine_mut()
                .client
                .reset_joint_state(body, joint, q, None)
                .map_err(|error| Failure::Device(error.to_string()))?;
        }
        MoveTo::<JointSpace<7>>::move_to(arm, HOME)
            .map_err(|error| Failure::Device(error.to_string()))
    }

    /// Where the simulator has the cube.
    pub fn cube_pose(arm: &mut Device, cube: i32) -> Result<Isometry3<f64>, Failure> {
        arm.engine_mut()
            .client
            .get_base_position_and_orientation(cube)
            .map_err(|error| Failure::Device(error.to_string()))
    }

    /// Puts the cube back on the table at `at` (x, y) and lets it settle:
    /// the reset between trials, done by the simulator, not by the arm.
    pub fn place_cube(arm: &mut Device, cube: i32, at: [f64; 2]) -> Result<(), Failure> {
        let device = |error: rsbullet::BulletError| Failure::Device(error.to_string());
        let client = &mut arm.engine_mut().client;
        client
            .reset_base_position_and_orientation(cube, [at[0], at[1], REST], [0.0, 0.0, 0.0, 1.0])
            .map_err(device)?;
        client
            .reset_base_velocity(cube, Some([0.0; 3]), Some([0.0; 3]))
            .map_err(device)?;
        for _ in 0..60 {
            arm.engine_mut().step().map_err(device)?;
        }
        Ok(())
    }
}

/// The table, the wall and a cube resting at `at`; returns the cube.
fn furnish(client: &mut PhysicsClient, at: [f64; 2]) -> anyhow::Result<i32> {
    let table = box_body(client, TABLE_HALF, TABLE_AT, 0.0)?;
    box_body(client, WALL_HALF, WALL_AT, 0.0)?;
    let cube = box_body(client, [CUBE / 2.0; 3], [at[0], at[1], REST], 0.05)?;
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
pub struct Planner {
    pub scene: PlanningScene<7>,
    grasp_point: Chain,
}

impl Planner {
    /// The scene with the `blocks` and the cube where `at` (x, y) says it
    /// rests.
    pub fn new(assets: &Path, at: [f64; 2], blocks: &[Block]) -> anyhow::Result<Self> {
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
        let place = |[x, y, z]: [f64; 3]| Isometry3::translation(x, y, z);
        scene.add_object("table", &block(TABLE_HALF), place(TABLE_AT))?;
        scene.add_object("wall", &block(WALL_HALF), place(WALL_AT))?;
        scene.add_object("cube", &block([CUBE / 2.0; 3]), place([at[0], at[1], REST]))?;
        for extra in blocks {
            scene.add_object(extra.name, &block(extra.half), place(extra.at))?;
        }
        scene.set_padding(PADDING)?;
        let mut planner = Self { scene, grasp_point };
        planner.fingers(OPEN)?;
        Ok(planner)
    }

    /// Sets each finger's opening (m) in the scene.
    pub fn fingers(&mut self, opening: f64) -> anyhow::Result<()> {
        for finger in FINGERS {
            self.scene.set_joint(finger, opening)?;
        }
        Ok(())
    }

    /// Joint positions placing the grasp point at `at`, pointing down and
    /// turned by `yaw`, starting the solver from `seed`.
    pub fn reach(&self, at: [f64; 3], yaw: Yaw, seed: [f64; 7]) -> Result<[f64; 7], Failure> {
        let target = Isometry3::from_parts(
            Translation3::from(at),
            UnitQuaternion::from_euler_angles(0.0, 0.0, yaw.0)
                * UnitQuaternion::from_euler_angles(PI, 0.0, 0.0),
        );
        self.reach_pose(target, seed)
    }

    /// Where the grasp point is at `q`.
    pub fn tip(&self, q: &[f64; 7]) -> Result<Isometry3<f64>, Failure> {
        self.grasp_point
            .forward_kinematics_iso(q)
            .map_err(|error| Failure::Plan(error.to_string()))
    }

    /// Joint positions placing the grasp point at `target`, starting the
    /// solver from `seed`.
    pub fn reach_pose(&self, target: Isometry3<f64>, seed: [f64; 7]) -> Result<[f64; 7], Failure> {
        let stop = CommonStop {
            pos_tol: 1e-5,
            rot_tol: 1e-4,
            max_iters: 1000,
            step_clip: 0.2,
        };
        let at = target.translation.vector;
        let unreachable = |why: String| {
            Failure::Unreachable(format!("({:.3}, {:.3}, {:.3}): {why}", at.x, at.y, at.z))
        };
        let solution = chain_ik(&self.grasp_point, &seed, &target.into(), 1e-3, stop)
            .map_err(|error| unreachable(error.to_string()))?;
        if !solution.converged {
            return Err(unreachable(format!(
                "IK stopped {:.4} m and {:.4} rad off",
                solution.position_error, solution.rotation_error
            )));
        }
        let q: [f64; 7] = std::array::from_fn(|i| solution.q[i]);
        if !(0..7).all(|i| (Panda::JOINT_MIN[i]..=Panda::JOINT_MAX[i]).contains(&q[i])) {
            return Err(unreachable(format!("IK left the joint limits: {q:?}")));
        }
        Ok(q)
    }

    /// A path from `start` to `goal` that is free in the scene as it is now.
    pub fn plan(&self, start: [f64; 7], goal: [f64; 7]) -> Result<Vec<[f64; 7]>, Failure> {
        let options = PlannerOptions {
            max_iterations: 20_000,
            seed: 20261029,
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
            Failure::Plan(format!(
                "{error}; at the start {:?}, at the goal {:?}",
                self.scene.first_contact(&start),
                self.scene.first_contact(&goal)
            ))
        })
    }
}

/// What one motion did.
#[derive(Clone, Copy, Debug, Default)]
pub struct Motion {
    /// Duration of the timed trajectory.
    pub planned: Duration,
    /// Largest distance (rad) between a commanded and the next measured
    /// position of any joint.
    pub tracking: f64,
}

/// The arm's measured joint positions.
pub fn joints(arm: &mut Device) -> Result<[f64; 7], Failure> {
    Arm::<7>::state(arm)
        .map_err(|error| Failure::Device(error.to_string()))?
        .joint
        .meas
        .q()
        .ok_or_else(|| Failure::Device("no measured joint positions".into()))
}

/// Times `path` with COPP at `speed` and builds the follower that plays it.
pub fn follower(
    path: &[[f64; 7]],
    speed: f64,
    clock: &Arc<ManualClock>,
) -> Result<Follower, Failure> {
    let samples = plan_waypoints_traj_via_copp::<Device, 7>(path, speed)
        .map_err(|error| Failure::Plan(error.to_string()))?;
    let period = Duration::from_secs_f64(<Device as Robot>::CONTROL_PERIOD);
    Ok(
        JointTrajectoryFollower::new(points_from_samples(&samples, period), clock.clone())
            .map_err(|error| Failure::Track(error.to_string()))?
            .with_path_tolerance([0.1; 7])
            .with_goal_tolerance([0.01; 7], Duration::from_secs(1)),
    )
}

/// Plans from where the arm is to `goal`, times the path at `speed` and plays
/// it on the joint position channel to the end.
pub fn move_to(
    arm: &mut Device,
    planner: &Planner,
    clock: &Arc<ManualClock>,
    goal: [f64; 7],
    speed: f64,
) -> Result<Motion, Failure> {
    let start = joints(arm)?;
    let path = planner.plan(start, goal)?;
    let mut follower = follower(&path, speed, clock)?;
    let planned = follower.duration();
    let mut tracking = 0.0_f64;
    let mut last: Option<[f64; 7]> = None;
    let mut refused = None;
    ControlWith::<JointPositionControl<7>>::control_with_flow(arm, |state, _period| {
        if let (Some(command), Some(q)) = (last, state.meas.q()) {
            for i in 0..7 {
                tracking = tracking.max((command[i] - q[i]).abs());
            }
        }
        match follower.step(&state) {
            Ok((command, done)) => {
                last = Some(command);
                ControlFlow::Continue((command, done))
            }
            Err(error) => {
                refused = Some(error);
                ControlFlow::Break(())
            }
        }
    })
    .map_err(|error| Failure::Device(error.to_string()))?;
    if let Some(error) = refused {
        return Err(Failure::Track(error.to_string()));
    }
    Ok(Motion { planned, tracking })
}

/// Closes on the cube or opens the hand.
pub fn grip(arm: &mut Device, close: bool) -> Result<f64, Failure> {
    let command = if close {
        GripperCommand {
            width: CUBE,
            speed: 0.05,
            force: 15.0,
        }
    } else {
        GripperCommand {
            width: 2.0 * OPEN,
            speed: 0.05,
            force: 0.0,
        }
    };
    let device = |error: roplat_behavior::RobotException| Failure::Device(error.to_string());
    arm.move_gripper(command).map_err(device)?;
    let state = arm.read_gripper().map_err(device)?;
    if close && state.grasped != 1 {
        return Err(Failure::Missed { width: state.width });
    }
    Ok(state.width)
}
