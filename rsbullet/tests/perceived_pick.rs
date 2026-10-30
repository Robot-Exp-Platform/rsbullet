//! Perception to planning on a Panda.
//!
//! Bullet renders a box on the table that the planning scene was not told
//! about. Its depth image, back-projected and with the table plane removed,
//! becomes voxels in the scene; `roplat_grasp::plan_pick` then plans picking
//! a cube with every waypoint clear of what the camera saw. Planning only:
//! nothing is executed, so the test runs in a debug build.

use std::path::PathBuf;

use nalgebra::{Isometry3, Matrix3, Point3, Rotation3, Translation3, UnitQuaternion, Vector3};
use roplat_behavior::msgs::{FrameId, Pose};
use roplat_description::{Geometry, RobotModel};
use roplat_grasp::{GraspOptions, PickOptions, grasps, plan_pick};
use roplat_kinematics::{IkSearch, chain_ik_solutions};
use roplat_scene::{CollisionCheck, EndEffector, PlanningScene, points::depth_to_points};
use rsbullet::{CameraSpec, Mode, RsBullet, UrdfOptions};

const PANDA: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../rsbullet-sys/bullet3/examples/pybullet/gym/pybullet_data/franka_panda"
);
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
const LOWER: [f64; 7] = [
    -2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973,
];
const UPPER: [f64; 7] = [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973];
const HOME: [f64; 7] = [0.0, -0.6, 0.0, -2.1, 0.0, 1.6, 0.8];

const TABLE_TOP: f64 = 0.15;
const CUBE: f64 = 0.04;
const CUBE_AT: [f64; 3] = [0.5, -0.15, TABLE_TOP + CUBE / 2.0];
/// The 0.2 m box the scene does not know about, standing on the table.
const OBSTACLE_AT: [f64; 3] = [0.45, 0.12, TABLE_TOP + 0.10];

fn asset(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/assets")
        .join(name)
}

fn at([x, y, z]: [f64; 3]) -> Isometry3<f64> {
    Isometry3::translation(x, y, z)
}

/// The Panda, the table and the cube: what the planner is told.
fn scene() -> PlanningScene<7> {
    let dir = PathBuf::from(PANDA);
    let model = RobotModel::from_urdf_file(dir.join("panda.urdf")).unwrap();
    let mut scene = PlanningScene::new(model, ARM, |name: &str| {
        dir.join(name.trim_start_matches("package://"))
    })
    .unwrap();
    for finger in FINGERS {
        scene.set_joint(finger, 0.04).unwrap();
    }
    scene
        .add_object(
            "table",
            &Geometry::Box { size: [0.7, 0.8, TABLE_TOP] },
            at([0.55, 0.0, TABLE_TOP / 2.0]),
        )
        .unwrap();
    scene
        .add_object("cube", &Geometry::Box { size: [CUBE; 3] }, at(CUBE_AT))
        .unwrap();
    scene
}

/// What the camera sees: the table top and the box (the Panda and the cube
/// are left out of the render).
fn rendered_points() -> Vec<[f64; 3]> {
    let mut sim = RsBullet::new(Mode::Direct).unwrap();
    let fixed = |pose| UrdfOptions { base: Some(pose), use_fixed_base: true, ..Default::default() };
    sim.client
        .load_urdf(
            asset("camera_slab.urdf"),
            Some(fixed(at([0.55, 0.0, TABLE_TOP - 0.01]))),
        )
        .unwrap();
    sim.client
        .load_urdf(asset("camera_block.urdf"), Some(fixed(at(OBSTACLE_AT))))
        .unwrap();
    // 1.2 m above the table centre, looking down; image right is world -y.
    let columns = Matrix3::from_columns(&[-Vector3::y(), -Vector3::x(), -Vector3::z()]);
    let rotation = UnitQuaternion::from_rotation_matrix(&Rotation3::from_matrix_unchecked(columns));
    let spec = CameraSpec {
        width: 96,
        height: 72,
        fov_y: 60f64.to_radians(),
        near: 0.1,
        far: 3.0,
        pose: Isometry3::from_parts(Translation3::new(0.5, 0.0, 1.2), rotation),
        frame: FrameId::new(1),
    };
    let frame = sim.render_rgbd(&spec).unwrap().value;
    depth_to_points(&frame.depth, &spec.info(), spec.near..=spec.far, 1)
        .into_iter()
        .map(|p| spec.pose * Point3::new(p[0], p[1], p[2]))
        // Remove the support plane, as a segmentation step would.
        .filter(|p| p.z > TABLE_TOP + 0.01)
        .map(|p| [p.x, p.y, p.z])
        .collect()
}

fn hand() -> EndEffector {
    EndEffector::new("hand", "panda_grasptarget", Isometry3::identity()).with_links([
        "panda_hand",
        "panda_leftfinger",
        "panda_rightfinger",
    ])
}

/// Inverse kinematics of the grasp target, nearest to the seed.
fn inverse(model: &RobotModel) -> impl FnMut(&Isometry3<f64>, &[f64; 7]) -> Option<[f64; 7]> {
    let chain = model.chain("panda_link0", "panda_grasptarget").unwrap();
    move |target, seed| {
        let search = IkSearch { restarts: 12, max_solutions: 1, ..IkSearch::default() };
        let solutions = chain_ik_solutions(
            &chain,
            &Pose::from_isometry(*target),
            seed,
            |_| true,
            search,
        )
        .ok()?;
        solutions.first().map(|q| std::array::from_fn(|i| q[i]))
    }
}

fn touches_seen(scene: &PlanningScene<7>, q: &[f64; 7]) -> bool {
    scene
        .contacts(q)
        .iter()
        .any(|c| c.first == "seen" || c.second == "seen")
}

#[test]
fn the_box_is_an_obstacle_only_once_seen() {
    let mut scene = scene();
    let model = scene.model().clone();
    // The hand inside the box, pointing down.
    let inside = Isometry3::from_parts(
        Translation3::new(OBSTACLE_AT[0], OBSTACLE_AT[1], OBSTACLE_AT[2] + 0.05),
        UnitQuaternion::from_euler_angles(std::f64::consts::PI, 0.0, 0.0),
    );
    let q = inverse(&model)(&inside, &HOME).expect("the box is within reach");
    assert!(
        scene.is_free(&q),
        "nothing told the scene about the box: {:?}",
        scene.first_contact(&q)
    );

    let points = rendered_points();
    assert!(
        points.len() > 200,
        "{} points above the table",
        points.len()
    );
    // Everything left is the box: within its 0.2 m extent (one voxel of slack).
    for p in &points {
        assert!(
            (0..3).all(|i| (p[i] - OBSTACLE_AT[i]).abs() < 0.12),
            "stray point {p:?}"
        );
    }
    let voxels = scene
        .add_points("seen", &points, 0.02, Isometry3::identity())
        .unwrap();
    assert!(voxels > 50, "{voxels} voxels");
    assert!(touches_seen(&scene, &q), "the seen box must block {q:?}");
}

#[test]
fn plan_pick_keeps_clear_of_the_seen_box() {
    let mut scene = scene();
    scene
        .add_points("seen", &rendered_points(), 0.02, Isometry3::identity())
        .unwrap();
    let model = scene.model().clone();
    let candidates = grasps(
        &Geometry::Box { size: [CUBE; 3] },
        &at(CUBE_AT),
        &GraspOptions::default(),
    );
    let plan = plan_pick(
        &scene,
        &hand(),
        "cube",
        Some("table"),
        &candidates,
        HOME,
        LOWER,
        UPPER,
        inverse(&model),
        PickOptions::default(),
    )
    .unwrap_or_else(|e| panic!("{e}"));
    for q in plan
        .to_pregrasp
        .iter()
        .chain(&plan.approach)
        .chain(&plan.lift)
    {
        assert!(
            !touches_seen(&scene, q),
            "waypoint {q:?} touches the seen box"
        );
    }
    // The hand ends 0.1 m above the grasp.
    let lifted = hand().pose(&scene, plan.lift.last().unwrap()).unwrap();
    assert!((lifted.translation.z - (plan.grasp.pose.translation.z + 0.1)).abs() < 1e-3);
}
