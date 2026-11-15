//! Worlds on different threads of one process, each with its own DIRECT
//! server.
//!
//! Bullet keeps the Wavefront OBJ meshes and the textures it has read in two
//! process-wide hash maps without a lock, and one search path for the whole
//! process. Before rsbullet-core took turns on the commands that use them,
//! threads building meshed worlds at once corrupted the heap: the process
//! aborted (`malloc(): unaligned tcache chunk detected`, `free(): invalid
//! pointer`), faulted in `b3HashMap<b3HashString, CachedObjResult>::insert`
//! under `create_multi_body`, or hung. And one client's search path replaced
//! another's.

use std::{
    path::Path,
    sync::{Arc, Barrier},
};

use nalgebra::Isometry3;
use rsbullet_core::{
    BulletError, CameraImageOptions, CollisionGeometry, CollisionId, Mode, MultiBodyBase,
    MultiBodyCreateOptions, PhysicsClient, PhysicsEngineParametersUpdate, Renderer, UrdfOptions,
    VisualGeometry, VisualId,
};

const DATA: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../rsbullet-sys/bullet3/examples/pybullet/gym/pybullet_data"
);
const ASSETS: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/assets");

const THREADS: usize = 8;
const ROUNDS: usize = 4;
const BOXES: usize = 4;

/// OBJ files Bullet reads only when a body that shows them is created.
const VISUALS: [&str; 8] = [
    "bunny.obj",
    "cube.obj",
    "sphere_smooth.obj",
    "teddy2_VHACD_CHs.obj",
    "terrain.obj",
    "textured_sphere_smooth.obj",
    "stone.obj",
    "cloth_z_up.obj",
];

/// What thread `index` builds: bodies showing `VISUALS`, starting at its own
/// index so that the threads read different files at once; a plane (an OBJ
/// visual); a duck from an OBJ collision mesh and an OBJ visual with a
/// texture; the Panda (OBJ meshes on every link); and a few boxes. Then it
/// steps. Returns the bodies and the Panda's joints.
fn build(index: usize, barrier: &Barrier) -> (i32, i32) {
    let mut client = PhysicsClient::connect(Mode::Direct).unwrap();
    client.set_gravity([0.0, 0.0, -9.81]).unwrap();
    client.set_additional_search_path(DATA).unwrap();
    barrier.wait();
    for k in 0..VISUALS.len() {
        let file = VISUALS[(index + k) % VISUALS.len()];
        let visual = client
            .create_visual_shape(
                &VisualGeometry::Mesh { file, scale: [0.1; 3] },
                None::<Isometry3<f64>>,
            )
            .unwrap();
        client
            .create_multi_body(&MultiBodyCreateOptions {
                base: MultiBodyBase {
                    visual_shape: VisualId(visual),
                    pose: Isometry3::translation(k as f64, -2.0, 0.5),
                    ..Default::default()
                },
                ..Default::default()
            })
            .unwrap();
    }
    client.load_urdf("plane.urdf", None::<UrdfOptions>).unwrap();
    let collision = client
        .create_collision_shape(
            &CollisionGeometry::MeshFile { file: "duck_vhacd.obj", scale: [1.0; 3] },
            None::<Isometry3<f64>>,
        )
        .unwrap();
    let visual = client
        .create_visual_shape(
            &VisualGeometry::Mesh { file: "duck.obj", scale: [1.0; 3] },
            None::<Isometry3<f64>>,
        )
        .unwrap();
    client
        .create_multi_body(&MultiBodyCreateOptions {
            base: MultiBodyBase {
                mass: 1.0,
                collision_shape: CollisionId(collision),
                visual_shape: VisualId(visual),
                pose: Isometry3::translation(0.0, 1.0, 0.5),
                ..Default::default()
            },
            ..Default::default()
        })
        .unwrap();
    let options = UrdfOptions { use_fixed_base: true, ..Default::default() };
    let panda = client
        .load_urdf("franka_panda/panda.urdf", Some(options))
        .unwrap();
    for i in 0..BOXES {
        let shape = client
            .create_collision_shape(
                &CollisionGeometry::Box { half_extents: [0.1; 3] },
                None::<Isometry3<f64>>,
            )
            .unwrap();
        client
            .create_multi_body(&MultiBodyCreateOptions {
                base: MultiBodyBase {
                    mass: 1.0,
                    collision_shape: CollisionId(shape),
                    pose: Isometry3::translation(1.0 + 0.3 * i as f64, 0.0, 0.5),
                    ..Default::default()
                },
                ..Default::default()
            })
            .unwrap();
    }
    for _ in 0..20 {
        client.step_simulation().unwrap();
    }
    (client.get_num_bodies(), client.get_num_joints(panda))
}

#[test]
fn eight_threads_build_meshed_worlds_at_once() {
    // Clears Bullet's mesh table between rounds, so that every round fills
    // it again from all threads at once.
    let mut keeper = PhysicsClient::connect(Mode::Direct).unwrap();
    for _round in 0..ROUNDS {
        for enable_file_caching in [false, true] {
            keeper
                .set_physics_engine_parameter(&PhysicsEngineParametersUpdate {
                    enable_file_caching: Some(enable_file_caching),
                    ..Default::default()
                })
                .unwrap();
        }
        let barrier = Arc::new(Barrier::new(THREADS));
        let threads: Vec<_> = (0..THREADS)
            .map(|index| {
                let barrier = barrier.clone();
                std::thread::spawn(move || build(index, &barrier))
            })
            .collect();
        for thread in threads {
            let built = thread.join().expect("a building thread panicked");
            assert_eq!(built, ((VISUALS.len() + 3 + BOXES) as i32, 12));
        }
    }
}

fn loads(client: &mut PhysicsClient, file: &str) -> bool {
    client.load_urdf(file, None::<UrdfOptions>).is_ok()
}

/// `plane.urdf` is only in `DATA`, `linear_axis.urdf` only in `ASSETS`.
#[test]
fn each_client_looks_files_up_on_its_own_search_path() {
    let mut data = PhysicsClient::connect(Mode::Direct).unwrap();
    data.set_additional_search_path(DATA).unwrap();
    let mut assets = PhysicsClient::connect(Mode::Direct).unwrap();
    assets.set_additional_search_path(ASSETS).unwrap();
    let mut none = PhysicsClient::connect(Mode::Direct).unwrap();
    assert!(Path::new(DATA).join("plane.urdf").exists());
    assert!(Path::new(ASSETS).join("linear_axis.urdf").exists());

    assert!(loads(&mut data, "plane.urdf"));
    assert!(!loads(&mut assets, "plane.urdf"));
    assert!(loads(&mut assets, "linear_axis.urdf"));
    assert!(!loads(&mut data, "linear_axis.urdf"));
    assert!(loads(&mut data, "plane.urdf"));
    assert!(!loads(&mut none, "plane.urdf"));
    assert!(!loads(&mut none, "linear_axis.urdf"));
    assert!(loads(&mut assets, "linear_axis.urdf"));
    assert_eq!(
        (
            data.get_num_bodies(),
            assets.get_num_bodies(),
            none.get_num_bodies()
        ),
        (2, 2, 0)
    );
}

/// Bullet reads a body's visual mesh when the body is created, looking it up
/// again by the name given to `create_visual_shape`; a mesh it does not find
/// leaves the body unseen.
#[test]
fn a_body_finds_its_visual_mesh_on_its_clients_search_path() {
    let mut data = PhysicsClient::connect(Mode::Direct).unwrap();
    data.set_additional_search_path(DATA).unwrap();
    let visual = data
        .create_visual_shape(
            &VisualGeometry::Mesh { file: "duck.obj", scale: [1.0; 3] },
            None::<Isometry3<f64>>,
        )
        .unwrap();
    let mut assets = PhysicsClient::connect(Mode::Direct).unwrap();
    assets.set_additional_search_path(ASSETS).unwrap();
    let duck = data
        .create_multi_body(&MultiBodyCreateOptions {
            base: MultiBodyBase { visual_shape: VisualId(visual), ..Default::default() },
            ..Default::default()
        })
        .unwrap();
    // The duck spans about x ∈ [-1, 0.7], y ∈ [0.1, 1.6], z ∈ [-0.5, 0.6].
    let frame = data
        .get_camera_image(&CameraImageOptions {
            width: 64,
            height: 64,
            view_matrix: Some(PhysicsClient::compute_view_matrix(
                [0.0, -3.0, 0.8],
                [0.0, 0.8, 0.0],
                [0.0, 0.0, 1.0],
            )),
            projection_matrix: Some(PhysicsClient::compute_projection_matrix_fov(
                60.0, 1.0, 0.1, 10.0,
            )),
            renderer: Some(Renderer::TinyRenderer),
            ..Default::default()
        })
        .unwrap();
    let seen = frame
        .segmentation_mask
        .iter()
        .filter(|&&id| id == duck)
        .count();
    assert!(seen > 100, "the duck covers {seen} of 4096 pixels");
}

#[test]
fn a_search_path_bullet_cannot_send_is_refused() {
    let mut client = PhysicsClient::connect(Mode::Direct).unwrap();
    client.set_additional_search_path(DATA).unwrap();
    let long = format!("{DATA}/{}", "x".repeat(1024));
    assert!(matches!(
        client.set_additional_search_path(&long),
        Err(BulletError::CommandFailed { code: -1, .. })
    ));
    assert!(loads(&mut client, "plane.urdf"));
}
