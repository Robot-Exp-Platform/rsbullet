//! A finite roplat graph drives a native Bullet Panda pick-and-place task.
//! Physics owns integration; graph nodes sense, plan, then queue one next-step command.
use nalgebra::{Isometry3, Translation3, UnitQuaternion};
use robot_behavior::behavior::*;
use robot_behavior::{AddRobot, EntityBuilder, PhysicsEngine, RobotDescription};
use roplat::{Lifecycle, Node, RoplatError};
use rsbullet::{
    BulletError, CameraImageOptions, CollisionGeometry, CollisionId, ContactPoint,
    ControlModeArray, DynamicsUpdate, InverseKinematicsOptions, JointType, LoadModelFlags, Mode,
    MultiBodyBase, MultiBodyCreateOptions, PhysicsClient, PhysicsEngineParametersUpdate, RsBullet,
    RsBulletRobot, SimRhythm, VisualGeometry, VisualId, VisualShapeOptions,
};
use serde::{Deserialize, Serialize};
use std::{
    fs::{self, File},
    io::{BufWriter, Write},
    path::PathBuf,
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

const STEPS: u64 = 5880;
const DT: f64 = 1.0 / 240.0;
const ARM_FORCE: [f64; 7] = [87., 87., 87., 87., 12., 12., 12.];
const FINGER_FORCE: [f64; 2] = [15., 15.];
const HOME: [f64; 7] = [0., -0.6, 0., -2.1, 0., 1.6, 0.8];
const TABLE_TOP: f64 = 0.15;

struct Panda;
impl RobotDescription for Panda {
    const URDF: Option<&'static str> = Some("franka_panda/panda.urdf");
}

#[derive(Clone, Serialize, Deserialize)]
struct Config {
    name: String,
    seed: u64,
    start: [f64; 3],
    goal: [f64; 3],
    steps: u64,
    dt_seconds: f64,
    cube_edge_m: f64,
    cube_mass_kg: f64,
    cube_friction: f64,
    finger_friction: f64,
    finger_force_n: [f64; 2],
    arm_force_nm: [f64; 7],
    self_collision_enabled: bool,
    robot_load_flags: i32,
    solver_iterations: usize,
    controller: String,
}
impl Config {
    fn named(name: &str) -> anyhow::Result<Self> {
        let (start, goal, seed) = match name {
            "a" | "a-repeat" => (
                [0.45, -0.08, TABLE_TOP + 0.021],
                [0.55, 0.10, TABLE_TOP + 0.02],
                2026102601,
            ),
            "b" => (
                [0.50, 0.07, TABLE_TOP + 0.021],
                [0.42, -0.12, TABLE_TOP + 0.02],
                2026102602,
            ),
            _ => anyhow::bail!("configuration must be a, b, or a-repeat"),
        };
        Ok(Self {
            name: name.into(),
            seed,
            start,
            goal,
            steps: STEPS,
            dt_seconds: Duration::from_secs_f64(DT).as_secs_f64(),
            cube_edge_m: 0.04,
            cube_mass_kg: 0.05,
            cube_friction: 0.8,
            finger_friction: 1.,
            finger_force_n: FINGER_FORCE,
            arm_force_nm: ARM_FORCE,
            self_collision_enabled: false,
            robot_load_flags: 0,
            solver_iterations: 100,
            controller: "minimum-jerk Cartesian stages; native damped IK; bounded position motors"
                .into(),
        })
    }
}

#[derive(Clone, Copy, Serialize)]
struct Command {
    phase: &'static str,
    target: [f64; 3],
    finger_position: f64,
}

#[derive(Clone, Serialize)]
struct Contact {
    body_a: i32,
    body_b: i32,
    link_a: i32,
    link_b: i32,
    position_a: [f64; 3],
    position_b: [f64; 3],
    normal_on_b: [f64; 3],
    distance: f64,
    normal_force: f64,
    friction_force_1: f64,
    friction_force_2: f64,
}
impl From<ContactPoint> for Contact {
    fn from(c: ContactPoint) -> Self {
        Self {
            body_a: c.body_a,
            body_b: c.body_b,
            link_a: c.link_a,
            link_b: c.link_b,
            position_a: c.position_on_a,
            position_b: c.position_on_b,
            normal_on_b: c.contact_normal_on_b,
            distance: c.contact_distance,
            normal_force: c.normal_force,
            friction_force_1: c.linear_friction_force_1,
            friction_force_2: c.linear_friction_force_2,
        }
    }
}
#[derive(Clone, Serialize)]
struct Frame {
    step: u64,
    simulated_seconds: f64,
    wall_seconds: f64,
    command: Command,
    object_position: [f64; 3],
    object_orientation_xyzw: [f64; 4],
    object_velocity: [f64; 6],
    ee_position: [f64; 3],
    joint_positions: Vec<f64>,
    joint_velocities: Vec<f64>,
    joint_motor_torques: Vec<f64>,
    left_finger_normal_force: f64,
    right_finger_normal_force: f64,
    object_contacts: Vec<Contact>,
    constraint_count: i32,
}
struct Evidence {
    active_command: Command,
    frames: Vec<Frame>,
    world: serde_json::Value,
    snapshots: Vec<(u64, &'static str, Vec<u8>)>,
}
type Shared = Arc<Mutex<Evidence>>;
type PreparedScene = (RsBullet, RsBulletRobot<Panda>, Shared, [f64; 3], [i32; 2]);

struct Sense {
    evidence: Shared,
}
impl Lifecycle for Sense {
    type Error = RoplatError;
}
impl Node for Sense {
    type Resources = (RsBulletRobot<Panda>,);
    type Input = ();
    type Output = Result<Frame, RoplatError>;
    async fn process<'call>(
        &'call mut self,
        _: (),
        (robot,): (&'call mut RsBulletRobot<Panda>,),
    ) -> Self::Output {
        // Exercise the high-level post-integration cache; world objects/contact evidence come
        // through the native observer bridge, outside the graph's same-cycle data edges.
        let _state = robot
            .read_state()
            .map_err(|e| RoplatError::Resource(e.to_string()))?;
        self.evidence
            .lock()
            .map_err(|_| RoplatError::Resource("evidence lock poisoned".into()))?
            .frames
            .last()
            .cloned()
            .ok_or_else(|| RoplatError::Resource("missing physics sample".into()))
    }
}

struct Planner {
    config: Config,
    origin: [f64; 3],
    grasp: [f64; 3],
}
impl Lifecycle for Planner {
    type Error = RoplatError;
}
fn blend(a: [f64; 3], b: [f64; 3], t: f64) -> [f64; 3] {
    let t = t.clamp(0., 1.);
    let s = t * t * t * (10. + t * (-15. + 6. * t));
    std::array::from_fn(|i| a[i] + (b[i] - a[i]) * s)
}
impl Node for Planner {
    type Resources = ();
    type Input = Frame;
    type Output = Command;
    async fn process(&mut self, f: Frame, _: ()) -> Command {
        let n = f.step;
        if n <= 240 {
            self.grasp = [
                f.object_position[0],
                f.object_position[1],
                f.object_position[2] + 0.002,
            ];
        }
        let grasp = self.grasp;
        let above = [grasp[0], grasp[1], grasp[2] + 0.18];
        let high_goal = [self.config.goal[0], self.config.goal[1], grasp[2] + 0.18];
        let place = [
            self.config.goal[0],
            self.config.goal[1],
            self.config.goal[2] + 0.002,
        ];
        let (phase, target, finger_position) = match n {
            0..=239 => ("settle", self.origin, 0.04),
            240..=959 => (
                "approach",
                blend(self.origin, above, (n - 240) as f64 / 720.),
                0.04,
            ),
            960..=1679 => (
                "descend",
                blend(above, grasp, (n - 960) as f64 / 720.),
                0.04,
            ),
            1680..=2159 => ("close", grasp, 0.0),
            2160..=2879 => ("lift", blend(grasp, above, (n - 2160) as f64 / 720.), 0.0),
            2880..=3839 => (
                "transport",
                blend(above, high_goal, (n - 2880) as f64 / 960.),
                0.0,
            ),
            3840..=4559 => (
                "lower",
                blend(high_goal, place, (n - 3840) as f64 / 720.),
                0.0,
            ),
            4560..=4919 => ("release", place, 0.04),
            4920..=5399 => (
                "retreat",
                blend(place, high_goal, (n - 4920) as f64 / 480.),
                0.04,
            ),
            _ => ("final_settle", high_goal, 0.04),
        };
        Command { phase, target, finger_position }
    }
}

struct Actuate {
    evidence: Shared,
    finger_joints: [i32; 2],
}
impl Lifecycle for Actuate {
    type Error = RoplatError;
}
impl Node for Actuate {
    type Resources = (RsBulletRobot<Panda>,);
    type Input = Command;
    type Output = Result<(), RoplatError>;
    async fn process<'call>(
        &'call mut self,
        command: Command,
        (robot,): (&'call mut RsBulletRobot<Panda>,),
    ) -> Self::Output {
        let body = robot.body_id;
        let arm = robot.joint_indices.clone();
        let fingers = self.finger_joints;
        let ee = robot.end_effector_link;
        let evidence = self.evidence.clone();
        robot
            .enqueue(move |client, _dt| {
                let mut joint_ids = arm.clone();
                joint_ids.extend(fingers);
                let seed: Vec<f64> = client
                    .get_joint_states(body, &joint_ids)?
                    .iter()
                    .map(|s| s.position)
                    .collect();
                let pose = Isometry3::from_parts(
                    Translation3::from(command.target),
                    UnitQuaternion::from_euler_angles(std::f64::consts::PI, 0., 0.),
                );
                let q = client.calculate_inverse_kinematics(
                    body,
                    ee,
                    pose,
                    &InverseKinematicsOptions {
                        current_positions: Some(&seed),
                        max_iterations: Some(100),
                        residual_threshold: Some(1e-5),
                        ..Default::default()
                    },
                )?;
                if q.len() < 7 || q.iter().any(|v| !v.is_finite()) {
                    return Err(BulletError::CommandFailed {
                        message: "invalid Panda IK result",
                        code: -1,
                    });
                }
                client.set_joint_motor_control_array(
                    body,
                    &arm,
                    ControlModeArray::Position(&q[..7]),
                    Some(&ARM_FORCE),
                )?;
                client.set_joint_motor_control_array(
                    body,
                    &fingers,
                    ControlModeArray::Position(&[command.finger_position; 2]),
                    Some(&FINGER_FORCE),
                )?;
                evidence
                    .lock()
                    .map_err(|_| BulletError::CommandFailed {
                        message: "evidence lock poisoned",
                        code: -1,
                    })?
                    .active_command = command;
                Ok(true)
            })
            .map_err(|e| RoplatError::Resource(e.to_string()))
    }
}

#[roplat::system]
async fn pick_and_place(
    mut simulation: SimRhythm<RsBullet>,
    mut robot: RsBulletRobot<Panda>,
    evidence: Shared,
    config: Config,
    initial_ee: [f64; 3],
    fingers: [i32; 2],
) -> roplat::RoplatResult<SimRhythm<RsBullet>> {
    let mut sense = Sense { evidence: evidence.clone() };
    let mut path = Planner { grasp: config.start, config, origin: initial_ee };
    let mut command = Actuate { evidence, finger_joints: fingers };
    simulation >> {
        &mut robot >> sense.resources.0;
        sense? >> path;
        &mut robot >> command.resources.0;
        path >> command;
        command?;
    };
    Ok(simulation)
}

fn box_body(
    client: &mut PhysicsClient,
    half: [f64; 3],
    position: [f64; 3],
    mass: f64,
) -> anyhow::Result<i32> {
    let shape = client.create_collision_shape(
        &CollisionGeometry::Box { half_extents: half },
        None::<Isometry3<f64>>,
    )?;
    let visual = client.create_visual_shape(
        &VisualGeometry::Box { half_extents: half },
        Some(VisualShapeOptions {
            rgba: if mass > 0. {
                [0.95, 0.3, 0.06, 1.]
            } else {
                [0.60, 0.65, 0.70, 1.]
            },
            ..Default::default()
        }),
    )?;
    Ok(client.create_multi_body(&MultiBodyCreateOptions {
        base: MultiBodyBase {
            mass,
            collision_shape: CollisionId(shape),
            visual_shape: VisualId(visual),
            pose: Isometry3::translation(position[0], position[1], position[2]),
            ..Default::default()
        },
        ..Default::default()
    })?)
}

fn prepare(
    config: &Config,
    assets: &str,
    started: Instant,
) -> anyhow::Result<PreparedScene> {
    let mut sim = RsBullet::new(Mode::Direct)?;
    sim.set_step_time(Duration::from_secs_f64(config.dt_seconds))?
        .set_gravity([0., 0., -9.81])?;
    sim.client.set_additional_search_path(assets)?;
    sim.client
        .set_physics_engine_parameter(&PhysicsEngineParametersUpdate {
            num_solver_iterations: Some(config.solver_iterations),
            deterministic_overlapping_pairs: Some(true),
            num_sub_steps: Some(0),
            ..Default::default()
        })?;
    let table = box_body(
        &mut sim.client,
        [0.35, 0.40, TABLE_TOP / 2.],
        [0.55, 0., TABLE_TOP / 2.],
        0.,
    )?;
    sim.client.change_dynamics(
        table,
        -1,
        &DynamicsUpdate {
            lateral_friction: Some(0.8),
            restitution: Some(0.),
            ..Default::default()
        },
    )?;
    let cube = box_body(
        &mut sim.client,
        [config.cube_edge_m / 2.; 3],
        config.start,
        config.cube_mass_kg,
    )?;
    sim.client.change_dynamics(
        cube,
        -1,
        &DynamicsUpdate {
            lateral_friction: Some(config.cube_friction),
            spinning_friction: Some(0.001),
            rolling_friction: Some(0.001),
            restitution: Some(0.),
            linear_damping: Some(0.04),
            angular_damping: Some(0.04),
            ..Default::default()
        },
    )?;
    let cube_dynamics = sim.client.get_dynamics_info(cube, -1)?;
    anyhow::ensure!(
        (cube_dynamics.mass - config.cube_mass_kg).abs() < 1e-6,
        "native cube mass {} differs from requested {}",
        cube_dynamics.mass,
        config.cube_mass_kg
    );
    anyhow::ensure!(
        (cube_dynamics.lateral_friction - config.cube_friction).abs() < 1e-6,
        "native cube friction {} differs from requested {}",
        cube_dynamics.lateral_friction,
        config.cube_friction
    );
    let robot = sim
        .robot_builder::<Panda>("panda")
        .joint_names([
            "panda_joint1",
            "panda_joint2",
            "panda_joint3",
            "panda_joint4",
            "panda_joint5",
            "panda_joint6",
            "panda_joint7",
        ])
        .end_effector_link_name("panda_grasptarget")
        .base_fixed(true)
        .flags(LoadModelFlags::empty())
        .load()?;
    let body = robot.body_id;
    let mut fingers = Vec::new();
    for i in 0..sim.client.get_num_joints(body) {
        let info = sim.client.get_joint_info(body, i)?;
        if info.joint_type == JointType::Prismatic {
            fingers.push(i);
        }
    }
    let fingers: [i32; 2] = fingers
        .try_into()
        .map_err(|_| anyhow::anyhow!("expected two Panda fingers"))?;
    for (&i, &q) in robot.joint_indices.iter().zip(&HOME) {
        sim.client.reset_joint_state(body, i, q, None)?;
    }
    for &i in &fingers {
        sim.client.reset_joint_state(body, i, 0.04, None)?;
        sim.client.change_dynamics(
            body,
            i,
            &DynamicsUpdate {
                lateral_friction: Some(config.finger_friction),
                spinning_friction: Some(0.1),
                friction_anchor: Some(true),
                ..Default::default()
            },
        )?;
    }
    sim.client.set_joint_motor_control_array(
        body,
        &robot.joint_indices,
        ControlModeArray::Position(&HOME),
        Some(&ARM_FORCE),
    )?;
    sim.client.set_joint_motor_control_array(
        body,
        &fingers,
        ControlModeArray::Position(&[0.04; 2]),
        Some(&FINGER_FORCE),
    )?;
    let ee = robot.end_effector_link;
    let initial_ee: [f64; 3] = sim
        .client
        .get_link_state(body, ee, true, true)?
        .world_link_frame
        .translation
        .vector
        .into();
    let evidence = Arc::new(Mutex::new(Evidence {
        active_command: Command { phase: "initial", target: initial_ee, finger_position: 0.04 },
        frames: Vec::with_capacity(config.steps as usize),
        snapshots: Vec::new(),
        world: serde_json::json!({
            "robot_body_id":body,"cube_body_id":cube,"table_body_id":table,
            "controlled_arm_joint_indices":robot.joint_indices,
            "controlled_arm_joint_names":robot.joint_names,
            "finger_joint_indices":fingers,"end_effector_link_index":ee,
            "cube_mass_readback_kg":cube_dynamics.mass,
            "cube_friction_readback":cube_dynamics.lateral_friction,
            "table_top_m":TABLE_TOP,"robot_load_flags":0,"self_collision_enabled":false,
            "created_runtime_constraints":0
        }),
    }));
    let observation = evidence.clone();
    let mut joints = robot.joint_indices.clone();
    joints.extend(fingers);
    let mut step = 0;
    let actual_dt = sim.step_time().as_secs_f64();
    sim.add_step_observer(move |client| {
        step += 1;
        let pose = client.get_base_position_and_orientation(cube)?;
        let q = pose.rotation.quaternion();
        let joints_state = client.get_joint_states(body, &joints)?;
        let contacts: Vec<Contact> = client
            .get_contact_points(Some(cube), None, None, None)?
            .into_iter()
            .map(Into::into)
            .collect();
        let force_on = |link| {
            contacts
                .iter()
                .filter(|c| {
                    (c.body_a == body && c.link_a == link) || (c.body_b == body && c.link_b == link)
                })
                .map(|c| c.normal_force)
                .sum()
        };
        let mut evidence = observation.lock().map_err(|_| BulletError::CommandFailed {
            message: "evidence lock poisoned",
            code: -1,
        })?;
        let frame = Frame {
            step,
            simulated_seconds: step as f64 * actual_dt,
            wall_seconds: started.elapsed().as_secs_f64(),
            command: evidence.active_command,
            object_position: pose.translation.vector.into(),
            object_orientation_xyzw: [q.i, q.j, q.k, q.w],
            object_velocity: client.get_base_velocity(cube)?,
            ee_position: client
                .get_link_state(body, ee, true, true)?
                .world_link_frame
                .translation
                .vector
                .into(),
            joint_positions: joints_state.iter().map(|s| s.position).collect(),
            joint_velocities: joints_state.iter().map(|s| s.velocity).collect(),
            joint_motor_torques: joints_state.iter().map(|s| s.motor_torque).collect(),
            left_finger_normal_force: force_on(fingers[0]),
            right_finger_normal_force: force_on(fingers[1]),
            object_contacts: contacts,
            constraint_count: client.get_num_constraints(),
        };
        if [240, 960, 1680, 2160, 2880, 3840, 4560, 4920, 5400, 5880].contains(&step) {
            let image = client.get_camera_image(&CameraImageOptions {
                width: 320,
                height: 240,
                view_matrix: Some(PhysicsClient::compute_view_matrix(
                    [1.0, -1.0, 0.85],
                    [0.45, 0., 0.28],
                    [0., 0., 1.],
                )),
                projection_matrix: Some(PhysicsClient::compute_projection_matrix_fov(
                    50.,
                    4. / 3.,
                    0.02,
                    3.,
                )),
                renderer: Some(rsbullet::Renderer::TinyRenderer),
                ..Default::default()
            })?;
            let mut ppm = format!("P6\n{} {}\n255\n", image.width, image.height).into_bytes();
            for pixel in image.rgba.chunks_exact(4) {
                ppm.extend_from_slice(&pixel[..3]);
            }
            evidence.snapshots.push((step, frame.command.phase, ppm));
        }
        evidence.frames.push(frame);
        Ok(())
    });
    Ok((sim, robot, evidence, initial_ee, fingers))
}

fn distance(a: [f64; 3], b: [f64; 3]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f64>()
        .sqrt()
}
fn horizontal(a: [f64; 3], b: [f64; 3]) -> f64 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt()
}

fn result(
    config: &Config,
    frames: &[Frame],
    graph_error: Option<String>,
    wall: f64,
) -> serde_json::Value {
    let Some(last) = frames.last() else {
        return serde_json::json!({"success":false,"graph_error":graph_error,"reason":"no integrated samples"});
    };
    let initial = config.start;
    let bilateral =
        |f: &&Frame| f.left_finger_normal_force > 0.05 && f.right_finger_normal_force > 0.05;
    let lifted: Vec<_> = frames
        .iter()
        .filter(|f| f.object_position[2] - initial[2] >= 0.10)
        .collect();
    let bilateral_lifted_steps = lifted.iter().copied().filter(bilateral).count();
    let max_lift = frames
        .iter()
        .map(|f| f.object_position[2] - initial[2])
        .fold(0., f64::max);
    let max_transport_while_lifted = lifted
        .iter()
        .map(|f| horizontal(f.object_position, initial))
        .fold(0., f64::max);
    let max_transport_with_bilateral_contact = lifted
        .iter()
        .copied()
        .filter(bilateral)
        .map(|f| horizontal(f.object_position, initial))
        .fold(0., f64::max);
    let final_error = distance(last.object_position, config.goal);
    let tail: Vec<_> = frames.iter().rev().take(240).collect();
    let max_tail_speed = tail
        .iter()
        .map(|f| {
            f.object_velocity[..3]
                .iter()
                .map(|x| x * x)
                .sum::<f64>()
                .sqrt()
        })
        .fold(0., f64::max);
    let max_tail_drift = tail
        .iter()
        .map(|f| distance(f.object_position, last.object_position))
        .fold(0., f64::max);
    let max_tail_angular_speed = tail
        .iter()
        .map(|f| {
            f.object_velocity[3..]
                .iter()
                .map(|x| x * x)
                .sum::<f64>()
                .sqrt()
        })
        .fold(0., f64::max);
    let final_released = tail
        .iter()
        .all(|f| f.left_finger_normal_force < 0.05 && f.right_finger_normal_force < 0.05);
    let force_caps = ARM_FORCE
        .into_iter()
        .chain(FINGER_FORCE)
        .collect::<Vec<_>>();
    let forces_bounded = frames.iter().all(|f| {
        f.joint_motor_torques
            .iter()
            .zip(&force_caps)
            .all(|(t, c)| t.is_finite() && t.abs() <= c + 1e-5)
    });
    let no_constraints = frames.iter().all(|f| f.constraint_count == 0);
    let settled = max_tail_speed < 0.01 && max_tail_angular_speed < 0.1 && max_tail_drift < 0.002;
    let success = graph_error.is_none()
        && frames.len() == config.steps as usize
        && max_lift >= 0.10
        && max_transport_while_lifted >= 0.12
        && max_transport_with_bilateral_contact >= 0.12
        && bilateral_lifted_steps >= 120
        && final_error <= 0.025
        && final_released
        && settled
        && forces_bounded
        && no_constraints;
    serde_json::json!({"success":success,"config":config,"backend":"native Bullet DIRECT", "python_or_pybullet_used":false,
        "graph_error":graph_error,"steps":frames.len(),"simulated_seconds":frames.len() as f64*config.dt_seconds,"run_wall_seconds":wall,
        "initial_object_position":initial,"final_object_position":last.object_position,"final_goal_error_m":final_error,
        "maximum_lift_m":max_lift,"maximum_horizontal_transport_while_lifted_m":max_transport_while_lifted,
        "maximum_horizontal_transport_while_lifted_with_bilateral_contact_m":max_transport_with_bilateral_contact,
        "bilateral_contact_while_lifted_steps":bilateral_lifted_steps,"bilateral_contact_while_lifted_seconds":bilateral_lifted_steps as f64 * config.dt_seconds,
        "final_released":final_released,"settled":settled,"tail_max_speed_m_s":max_tail_speed,"tail_max_angular_speed_rad_s":max_tail_angular_speed,"tail_max_drift_m":max_tail_drift,
        "all_motor_efforts_within_explicit_caps":forces_bounded,"no_runtime_constraints":no_constraints,
        "object_pose_writes_after_initialization":0,"hidden_attachment":false,
        "collision_scope":"Panda self-collision disabled (flags=0); robot-object, robot-table and object-table enabled; no self-collision fidelity claim"})
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> anyhow::Result<()> {
    let task_started = Instant::now();
    let args: Vec<String> = std::env::args().collect();
    anyhow::ensure!(
        args.len() == 4,
        "usage: panda_pick_place a|b|a-repeat MODEL_DIRECTORY OUTPUT_DIRECTORY"
    );
    let config = Config::named(&args[1])?;
    let output = PathBuf::from(&args[3]);
    if let Some(parent) = output
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        fs::create_dir_all(parent)?;
    }
    fs::create_dir(&output)?;
    fs::write(
        output.join("config.json"),
        serde_json::to_vec_pretty(&config)?,
    )?;
    let (sim, robot, evidence, initial_ee, fingers) = prepare(&config, &args[2], task_started)?;
    let graph = pick_and_place(
        SimRhythm::logical(sim).with_max_steps(config.steps),
        robot,
        evidence.clone(),
        config.clone(),
        initial_ee,
        fingers,
    )
    .await;
    let graph_error = match graph {
        Ok(simulation) => {
            simulation.into_engine().shutdown();
            None
        }
        Err(error) => Some(error.to_string()),
    };
    let elapsed = task_started.elapsed().as_secs_f64();
    let evidence = evidence
        .lock()
        .map_err(|_| anyhow::anyhow!("evidence lock poisoned"))?;
    let summary = result(&config, &evidence.frames, graph_error, elapsed);
    fs::write(
        output.join("world.json"),
        serde_json::to_vec_pretty(&evidence.world)?,
    )?;
    for (step, phase, ppm) in &evidence.snapshots {
        fs::write(output.join(format!("{step:05}-{phase}.ppm")), ppm)?;
    }
    let mut trace = BufWriter::new(File::create(output.join("trace.jsonl"))?);
    for frame in &evidence.frames {
        serde_json::to_writer(&mut trace, frame)?;
        writeln!(trace)?;
    }
    trace.flush()?;
    fs::write(
        output.join("summary.json"),
        serde_json::to_vec_pretty(&summary)?,
    )?;
    println!("{}", serde_json::to_string_pretty(&summary)?);
    if summary["success"] != true {
        std::process::exit(2);
    }
    Ok(())
}
