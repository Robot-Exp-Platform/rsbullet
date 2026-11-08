//! The rendered image is the one `CameraSpec::info` describes: every pixel
//! sees along the ray its intrinsics give it, with the principal point
//! anywhere, unequal focal lengths and lens distortion.
//!
//! A wall tilted about the optical frame's y axis has a depth that tells
//! each pixel's undistorted `x`, and one tilted about x its `y`. The two
//! renders give every pixel's ray, which is projected back through the
//! intrinsics, the `CameraInfo` model written out here and not taken from
//! the renderer, to land on the pixel itself.

use std::f64::consts::PI;
use std::path::PathBuf;

use nalgebra::{Isometry3, Translation3, Unit, UnitQuaternion, Vector3};
use roplat_behavior::msgs::{FrameId, distortion_model};
use rsbullet::{CameraSpec, Distortion, Intrinsics, Mode, RsBullet, UrdfOptions};

const WIDTH: u32 = 64;
const HEIGHT: u32 = 48;
/// The wall's distance along the optical axis (m) and its tilt (rad).
const Z0: f64 = 1.0;
const TILT: f64 = PI / 6.0;

fn asset(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/assets")
        .join(name)
}

/// A camera at the world origin, its optical frame the world's.
fn spec(intrinsics: Intrinsics) -> CameraSpec {
    CameraSpec {
        width: WIDTH,
        height: HEIGHT,
        intrinsics,
        near: 0.1,
        far: 3.0,
        pose: Isometry3::identity(),
        frame: FrameId::new(1),
        depth_noise: None,
    }
}

/// The depth of the wall, its near face through `(0, 0, Z0)` and turned by
/// `TILT` about `axis`.
fn wall_depth(intrinsics: Intrinsics, axis: Vector3<f64>) -> Vec<f32> {
    let mut sim = RsBullet::new(Mode::Direct).unwrap();
    let turn = UnitQuaternion::from_axis_angle(&Unit::new_normalize(axis), TILT);
    let centre = Vector3::new(0.0, 0.0, Z0) + 0.01 * (turn * Vector3::z());
    let options = UrdfOptions {
        base: Some(Isometry3::from_parts(Translation3::from(centre), turn)),
        use_fixed_base: true,
        ..Default::default()
    };
    sim.client
        .load_urdf(asset("camera_wall.urdf"), Some(options))
        .unwrap();
    sim.render_rgbd(&spec(intrinsics)).unwrap().value.depth
}

/// Every pixel's undistorted normalised coordinates `(x, y)`, row by row.
/// Turned about y the wall's near face is `z = Z0 / (1 + x tan TILT)` along
/// the ray `(x, y, 1)`; turned about x, `z = Z0 / (1 - y tan TILT)`.
fn rays(intrinsics: Intrinsics) -> Vec<[f64; 2]> {
    let about_y = wall_depth(intrinsics, Vector3::y());
    let about_x = wall_depth(intrinsics, Vector3::x());
    let t = TILT.tan();
    about_y
        .iter()
        .zip(&about_x)
        .map(|(&zy, &zx)| {
            assert!(zy.is_finite() && zx.is_finite(), "the wall fills the view");
            [
                (Z0 / f64::from(zy) - 1.0) / t,
                (1.0 - Z0 / f64::from(zx)) / t,
            ]
        })
        .collect()
}

/// The pixel a ray lands on: the distortion models of `CameraInfo`.
fn project(k: &Intrinsics, [x, y]: [f64; 2]) -> [f64; 2] {
    let [k1, k2, p1, p2, k3, k4, k5, k6] = match k.distortion {
        Distortion::None => [0.0; 8],
        Distortion::PlumbBob([k1, k2, p1, p2, k3]) => [k1, k2, p1, p2, k3, 0.0, 0.0, 0.0],
        Distortion::RationalPolynomial(d) => d,
    };
    let r2 = x * x + y * y;
    let (r4, r6) = (r2 * r2, r2 * r2 * r2);
    let radial = (1.0 + k1 * r2 + k2 * r4 + k3 * r6) / (1.0 + k4 * r2 + k5 * r4 + k6 * r6);
    let xd = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x);
    let yd = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y;
    [k.fx * xd + k.cx, k.fy * yd + k.cy]
}

/// The largest and the mean distance (px) from a pixel to where its ray
/// lands by `model`.
fn misplacement(rays: &[[f64; 2]], model: &Intrinsics) -> (f64, f64) {
    let width = WIDTH as usize;
    let (mut worst, mut sum) = (0.0_f64, 0.0);
    for (i, ray) in rays.iter().enumerate() {
        let (u, v) = ((i % width) as f64, (i / width) as f64);
        let [pu, pv] = project(model, *ray);
        let off = (pu - u).hypot(pv - v);
        worst = worst.max(off);
        sum += off;
    }
    (worst, sum / rays.len() as f64)
}

#[test]
fn a_centred_pinhole_sees_along_the_rays_of_its_pixel_centres() {
    let intrinsics = Intrinsics::centred(WIDTH, HEIGHT, 60f64.to_radians());
    let info = spec(intrinsics).info();
    assert_eq!(
        info.k,
        [
            intrinsics.fx,
            0.0,
            31.5,
            0.0,
            intrinsics.fy,
            23.5,
            0.0,
            0.0,
            1.0
        ]
    );
    assert_eq!(info.distortion_model, distortion_model::NONE);
    // Before, the image sat half a pixel right of and above this.
    let (worst, _) = misplacement(&rays(intrinsics), &intrinsics);
    assert!(worst < 0.02, "a pixel's ray lands {worst} px from it");
}

#[test]
fn focal_lengths_and_the_principal_point_are_the_rendered_ones() {
    let intrinsics = Intrinsics {
        fx: 58.0,
        fy: 44.0,
        cx: 36.3,
        cy: 20.7,
        distortion: Distortion::None,
    };
    let info = spec(intrinsics).info();
    assert_eq!(info.k, [58.0, 0.0, 36.3, 0.0, 44.0, 20.7, 0.0, 0.0, 1.0]);
    let (worst, _) = misplacement(&rays(intrinsics), &intrinsics);
    assert!(worst < 0.02, "a pixel's ray lands {worst} px from it");
}

/// Rendering a distorted image resamples a finer undistorted one at the
/// nearest pixel, within about a quarter of a pixel.
fn distorted(distortion: Distortion, model: u32, d: [f64; 8]) {
    let intrinsics = Intrinsics { fx: 50.0, fy: 50.0, cx: 31.5, cy: 23.5, distortion };
    let info = spec(intrinsics).info();
    assert_eq!((info.distortion_model, info.d), (model, d));
    let rays = rays(intrinsics);
    let (worst, mean) = misplacement(&rays, &intrinsics);
    assert!(
        worst < 0.45 && mean < 0.25,
        "rays land up to {worst} px, on average {mean} px, from their pixels"
    );
    // The lens is strong enough to tell: a pinhole of the same intrinsics
    // would put the corners' rays pixels away.
    let pinhole = Intrinsics { distortion: Distortion::None, ..intrinsics };
    let (worst, _) = misplacement(&rays, &pinhole);
    assert!(worst > 3.0, "the distortion moves pixels only {worst} px");
}

#[test]
fn a_plumb_bob_lens_bends_the_rays_as_camera_info_says() {
    let d = [-0.25, 0.06, 1e-3, -5e-4, 0.01];
    distorted(
        Distortion::PlumbBob(d),
        distortion_model::PLUMB_BOB,
        [d[0], d[1], d[2], d[3], d[4], 0.0, 0.0, 0.0],
    );
}

#[test]
fn a_rational_lens_bends_the_rays_as_camera_info_says() {
    let d = [0.4, -0.1, -1e-3, 2e-3, 0.0, 0.7, -0.05, 0.0];
    distorted(
        Distortion::RationalPolynomial(d),
        distortion_model::RATIONAL_POLYNOMIAL,
        d,
    );
}
