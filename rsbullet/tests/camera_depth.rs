//! Depth of the simulated camera as the `Camera` contract has it: one
//! missing value, infinity, for nothing within range, never a finite number
//! beyond the far plane; and seeded noise of the configured size.

use std::f64::consts::PI;
use std::path::PathBuf;

use nalgebra::{Isometry3, Translation3, UnitQuaternion, Vector3};
use roplat_behavior::Camera;
use roplat_behavior::conformance::{FacingWall, FrameView, check_camera, check_camera_facing_wall};
use roplat_behavior::msgs::FrameId;
use roplat_world::PhysicsEngine;
use rsbullet::{CameraSpec, DepthNoise, Intrinsics, Mode, RgbdFrame, RsBullet, UrdfOptions};

const WIDTH: u32 = 64;
const HEIGHT: u32 = 48;

fn asset(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/assets")
        .join(name)
}

/// A wall whose near face crosses the optical axis `distance` ahead of a
/// camera at the origin looking along +z, turned by `tilt` about y.
fn wall(distance: f64, tilt: f64) -> RsBullet {
    let mut sim = RsBullet::new(Mode::Direct).unwrap();
    let turn = UnitQuaternion::from_axis_angle(&Vector3::y_axis(), tilt);
    let centre = Vector3::new(0.0, 0.0, distance) + 0.01 * (turn * Vector3::z());
    let options = UrdfOptions {
        base: Some(Isometry3::from_parts(Translation3::from(centre), turn)),
        use_fixed_base: true,
        ..Default::default()
    };
    sim.client
        .load_urdf(asset("camera_wall.urdf"), Some(options))
        .unwrap();
    sim
}

fn spec(far: f64, depth_noise: Option<DepthNoise>) -> CameraSpec {
    CameraSpec {
        width: WIDTH,
        height: HEIGHT,
        intrinsics: Intrinsics::centred(WIDTH, HEIGHT, 60f64.to_radians()),
        near: 0.1,
        far,
        pose: Isometry3::identity(),
        frame: FrameId::new(2),
        depth_noise,
    }
}

/// 2 mm plus 4 mm per metre, one pixel in twenty lost.
const NOISE: DepthNoise = DepthNoise { sigma: [0.002, 0.004, 0.0], dropout: 0.05, seed: 42 };

fn view(frame: &RgbdFrame) -> FrameView {
    FrameView {
        width: frame.width,
        height: frame.height,
        depth: frame
            .depth
            .iter()
            .map(|&d| d.is_finite().then_some(f64::from(d)))
            .collect(),
    }
}

#[test]
fn a_facing_wall_reads_its_distance_in_every_pixel() {
    let mut sim = wall(1.0, 0.0);
    let mut camera = sim.camera(spec(3.0, None));
    let facing = FacingWall { distance: 1.0, tolerance: 1e-4, coverage: 1.0 };
    check_camera_facing_wall(&mut camera, || {}, 3, view, facing).unwrap();
}

#[test]
fn noise_has_the_configured_spread_and_loses_the_configured_share() {
    let mut sim = wall(1.0, 0.0);
    let frame = sim.render_rgbd(&spec(3.0, Some(NOISE))).unwrap().value;
    let read: Vec<f64> = frame
        .depth
        .iter()
        .filter(|d| d.is_finite())
        .map(|&d| f64::from(d))
        .collect();
    let pixels = (WIDTH * HEIGHT) as f64;
    let lost = 1.0 - read.len() as f64 / pixels;
    assert!((0.035..0.065).contains(&lost), "{lost} of the pixels lost");
    let n = read.len() as f64;
    let mean = read.iter().sum::<f64>() / n;
    let sigma = (read.iter().map(|d| (d - mean).powi(2)).sum::<f64>() / (n - 1.0)).sqrt();
    // At 1 m: 2 mm + 4 mm.
    assert!((mean - 1.0).abs() < 5e-4, "mean depth {mean} m");
    assert!(
        (sigma - 0.006).abs() < 0.0003,
        "standard deviation {sigma} m"
    );
}

#[test]
fn noise_is_reproduced_from_its_seed_and_the_frame_time() {
    let mut sim = wall(1.0, 0.0);
    let render = |sim: &mut RsBullet, noise: DepthNoise| {
        sim.render_rgbd(&spec(3.0, Some(noise)))
            .unwrap()
            .value
            .depth
    };
    let first = render(&mut sim, NOISE);
    assert_eq!(render(&mut sim, NOISE), first, "the same seed and time");
    assert_eq!(render(&mut wall(1.0, 0.0), NOISE), first, "another world");
    assert_ne!(render(&mut sim, DepthNoise { seed: 43, ..NOISE }), first);
    sim.step().unwrap();
    assert_ne!(render(&mut sim, NOISE), first, "a later frame");
}

#[test]
fn depth_beyond_the_far_plane_is_missing() {
    // Turned 30 degrees, the wall recedes from 0.73 m on the image's right
    // to 1.59 m on its left; beyond 1.2 m, left of about column 19.5,
    // nothing is rendered.
    let mut sim = wall(1.0, PI / 6.0);
    for noise in [None, Some(DepthNoise { sigma: [0.01, 0.0, 0.0], ..NOISE })] {
        let spec = spec(1.2, noise);
        let frame = sim.render_rgbd(&spec).unwrap().value;
        for v in 0..HEIGHT {
            for u in 0..WIDTH {
                let d = frame.depth_at(u, v);
                if d.is_finite() {
                    assert!(
                        (spec.near..=spec.far).contains(&f64::from(d)),
                        "({u}, {v}) reads {d} m"
                    );
                }
                if u <= 18 {
                    assert!(
                        d.is_infinite(),
                        "({u}, {v}) beyond the far plane reads {d} m"
                    );
                }
                if noise.is_none() && u >= 21 {
                    assert!(d.is_finite(), "({u}, {v}) within range reads nothing");
                }
            }
        }
    }
}

#[test]
fn a_noisy_camera_passes_the_contract_checks() {
    let mut sim = wall(1.0, 0.0);
    let mut camera = sim.camera(spec(3.0, Some(NOISE)));
    check_camera(&mut camera, || {}, 3, view).unwrap();
    // Within five standard deviations, at least nine in ten pixels read.
    let facing = FacingWall { distance: 1.0, tolerance: 0.03, coverage: 0.9 };
    check_camera_facing_wall(&mut camera, || {}, 3, view, facing).unwrap();
    assert_eq!(camera.capture().unwrap().header.seq, 7);
}
