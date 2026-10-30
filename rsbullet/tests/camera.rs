//! The simulated camera sees a block on a slab where it is.

use std::path::PathBuf;

use nalgebra::{Isometry3, Matrix3, Point3, Rotation3, Translation3, UnitQuaternion, Vector3};
use roplat_behavior::{Camera, msgs::FrameId};
use roplat_scene::points::depth_to_points;
use rsbullet::{CameraSpec, Mode, RsBullet, UrdfOptions};

fn asset(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/assets")
        .join(name)
}

/// The slab's top at z = 0.01, and a 0.2 m block on it centred at
/// (0.25, 0, 0.11), its top at z = 0.21.
fn world() -> RsBullet {
    let mut sim = RsBullet::new(Mode::Direct).unwrap();
    let place = |x: f64, z: f64| UrdfOptions {
        base: Some(Isometry3::translation(x, 0.0, z)),
        use_fixed_base: true,
        ..Default::default()
    };
    sim.client
        .load_urdf(asset("camera_slab.urdf"), Some(place(0.0, 0.0)))
        .unwrap();
    sim.client
        .load_urdf(asset("camera_block.urdf"), Some(place(0.25, 0.11)))
        .unwrap();
    sim
}

/// 1 m above the origin looking straight down. Image right is world -y and
/// image down is world -x, so the block (towards +x) is in the upper half.
fn looking_down() -> CameraSpec {
    let columns = Matrix3::from_columns(&[-Vector3::y(), -Vector3::x(), -Vector3::z()]);
    let rotation = UnitQuaternion::from_rotation_matrix(&Rotation3::from_matrix_unchecked(columns));
    CameraSpec {
        width: 64,
        height: 48,
        fov_y: 60f64.to_radians(),
        near: 0.1,
        far: 3.0,
        pose: Isometry3::from_parts(Translation3::new(0.0, 0.0, 1.0), rotation),
        frame: FrameId::new(3),
    }
}

#[test]
fn depth_is_metric_and_the_image_is_upright() {
    let mut sim = world();
    let spec = looking_down();
    let frame = sim.render_rgbd(&spec).unwrap().value;
    assert_eq!((frame.width, frame.height), (64, 48));
    assert_eq!(frame.rgba.len(), 64 * 48 * 4);

    // The block's top centre, (0.25, 0, 0.21), is 0.79 m along the view
    // and 0.25 m towards image-up: v = cy - f * 0.25 / 0.79.
    let info = spec.info();
    let (f, cx, cy) = (info.k[0], info.k[2], info.k[5]);
    let v_block = (cy - f * 0.25 / 0.79).round() as u32;
    let v_mirror = (cy + f * 0.25 / 0.79).round() as u32;
    let u = cx.round() as u32;
    let block = frame.depth_at(u, v_block);
    let slab = frame.depth_at(u, v_mirror);
    assert!(
        (block - 0.79).abs() < 0.02,
        "block top reads {block} m at v = {v_block}"
    );
    assert!(
        (slab - 0.99).abs() < 0.02,
        "slab reads {slab} m at v = {v_mirror}"
    );
    // Straight down, between them, the slab.
    assert!((frame.depth_at(u, cy.round() as u32) - 0.99).abs() < 0.02);
}

#[test]
fn back_projected_points_land_on_the_block_and_the_slab() {
    let mut sim = world();
    let spec = looking_down();
    let frame = sim.render_rgbd(&spec).unwrap().value;
    let points = depth_to_points(&frame.depth, &spec.info(), spec.near..=spec.far, 1);
    assert!(
        points.len() > 64 * 48 * 9 / 10,
        "{} of {} pixels hit",
        points.len(),
        64 * 48
    );
    let world: Vec<Point3<f64>> = points
        .iter()
        .map(|p| spec.pose * Point3::new(p[0], p[1], p[2]))
        .collect();
    // Points over the block's footprint are on its top, the rest on the slab.
    for p in &world {
        let over_block = (p.x - 0.25).abs() < 0.09 && p.y.abs() < 0.09;
        let expected = if over_block { 0.21 } else { 0.01 };
        let on_side = (p.x - 0.25).abs() < 0.11 && p.y.abs() < 0.11;
        if on_side && !over_block {
            continue; // the block's edges: either height
        }
        assert!(
            (p.z - expected).abs() < 0.02,
            "point {p:?} expected at z = {expected}"
        );
    }
}

#[test]
fn the_camera_contract_numbers_frames_and_reports_intrinsics() {
    let mut sim = world();
    let spec = looking_down();
    let mut camera = sim.camera(spec);
    let info = camera.camera_info();
    assert_eq!((info.width, info.height), (64, 48));
    // 24 px / tan(30 deg).
    assert!((info.k[0] - 24.0 / 30f64.to_radians().tan()).abs() < 1e-9);
    let first = camera.capture().unwrap();
    let second = camera.capture().unwrap();
    assert_eq!((first.header.seq, second.header.seq), (1, 2));
    assert_eq!(first.header.frame, FrameId::new(3));
    // Nothing above the near plane is missed: looking up, nothing is hit.
    camera.set_pose(Isometry3::from_parts(
        Translation3::new(0.0, 0.0, 1.0),
        UnitQuaternion::from_axis_angle(&Vector3::x_axis(), std::f64::consts::PI)
            * spec.pose.rotation,
    ));
    let up = camera.capture().unwrap().value;
    assert!(up.depth.iter().all(|d| d.is_infinite()));
}
