//! A simulated RGB-D camera behind the shared `Camera` contract.

use std::f64::consts::TAU;

use nalgebra as na;
use roplat_behavior::{
    Camera, RobotException, RobotResult,
    msgs::{CameraInfo, FrameId, Header, Stamp, Stamped, clock, distortion_model},
};
use rsbullet_core::{BulletError, BulletResult, CameraImageOptions, PhysicsClient, Renderer};

use crate::RsBullet;

/// Normalised depth this close to 1 is the far plane: no hit.
const NO_HIT: f64 = 1e-6;

/// Pixels of the undistorted image a distorted camera renders per pixel of
/// its own, along each axis, before it resamples: the nearest rendered
/// pixel is then within about a quarter of a pixel of the one sought.
const OVERSAMPLING: f64 = 2.0;

/// The longest side of the undistorted image a distorted camera renders.
const MAX_RENDER_SIDE: f64 = 8192.0;

/// Iterations allowed for undistorting one pixel, and how close (in
/// normalised image coordinates) the result must distort back to it.
const UNDISTORT_ITERATIONS: usize = 100;
const UNDISTORT_RESIDUAL: f64 = 1e-12;

/// Where a simulated camera is and what it sees: a camera rendered by
/// Bullet's software renderer, which also works without a display
/// ([`Mode::Direct`](crate::Mode::Direct)).
///
/// The rendered image is the one [`info`](Self::info) describes, to a
/// small fraction of a pixel without distortion and to about a quarter of
/// a pixel with it (see [`Distortion`]).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CameraSpec {
    pub width: u32,
    pub height: u32,
    /// The projection onto the image: focal lengths, principal point and
    /// lens distortion.
    pub intrinsics: Intrinsics,
    /// Nearest and farthest distances along the optical axis rendered (m).
    pub near: f64,
    pub far: f64,
    /// The optical frame in the world: `z` along the view, `x` right and
    /// `y` down in the image, as `CameraInfo` and depth back-projection
    /// expect.
    pub pose: na::Isometry3<f64>,
    /// The optical frame named in each frame's header.
    pub frame: FrameId,
    /// Noise on the rendered depth, or `None` for the exact depth.
    pub depth_noise: Option<DepthNoise>,
}

/// A pinhole projection in pixels, with lens distortion.
///
/// A point `(x, y, z)` in the optical frame lands, before distortion, at
/// `u = fx x / z + cx` and `v = fy y / z + cy`, with pixel `(u, v)` centred
/// on whole coordinates, column 0 at the left and row 0 at the top: the
/// convention of `CameraInfo::k`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Intrinsics {
    pub fx: f64,
    pub fy: f64,
    pub cx: f64,
    pub cy: f64,
    pub distortion: Distortion,
}

impl Intrinsics {
    /// Square pixels, the principal point at the centre of a `width` by
    /// `height` image, `((width - 1) / 2, (height - 1) / 2)`, a vertical
    /// field of view of `fov_y` (rad), and no distortion.
    pub fn centred(width: u32, height: u32, fov_y: f64) -> Self {
        let f = f64::from(height) / 2.0 / (fov_y / 2.0).tan();
        Self {
            fx: f,
            fy: f,
            cx: (f64::from(width) - 1.0) / 2.0,
            cy: (f64::from(height) - 1.0) / 2.0,
            distortion: Distortion::None,
        }
    }
}

/// Lens distortion in the models of `CameraInfo`, with its coefficients in
/// the order of `CameraInfo::d`.
///
/// With undistorted normalised coordinates `(x, y)` and `r² = x² + y²`, a
/// point lands at `(x', y')` with
/// `x' = x s + 2 p1 x y + p2 (r² + 2 x²)` and
/// `y' = y s + p1 (r² + 2 y²) + 2 p2 x y`, where
/// `s = (1 + k1 r² + k2 r⁴ + k3 r⁶) / (1 + k4 r² + k5 r⁴ + k6 r⁶)`.
///
/// A distorted camera renders an undistorted image twice as fine that
/// covers every pixel's ray, and takes each pixel's colour and depth from
/// the rendered pixel nearest its ray; the depth stays the point's `z`. Rendering fails where the
/// distortion cannot be inverted within the image. The fisheye model
/// (`EQUIDISTANT`) is not offered: its field can exceed what one pinhole
/// image covers.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Distortion {
    None,
    /// `[k1, k2, p1, p2, k3]`.
    PlumbBob([f64; 5]),
    /// `[k1, k2, p1, p2, k3, k4, k5, k6]`.
    RationalPolynomial([f64; 8]),
}

impl Distortion {
    /// The `CameraInfo` model and coefficients.
    fn camera_info(self) -> (u32, [f64; 8]) {
        match self {
            Self::None => (distortion_model::NONE, [0.0; 8]),
            Self::PlumbBob([k1, k2, p1, p2, k3]) => (
                distortion_model::PLUMB_BOB,
                [k1, k2, p1, p2, k3, 0.0, 0.0, 0.0],
            ),
            Self::RationalPolynomial(d) => (distortion_model::RATIONAL_POLYNOMIAL, d),
        }
    }
}

/// Noise on rendered depth: Gaussian with a standard deviation that grows
/// with the depth, and pixels lost at random, both reproducible.
///
/// The noise of a pixel is drawn from `seed`, the frame's simulated time
/// and the pixel's index alone: the same seed renders the same noise into
/// a frame of the same instant, whatever was rendered before, and two
/// frames of one instant carry the same noise. Give each camera its own
/// seed. A depth that the noise moves outside `near..=far` is missing, as a
/// device reports a reading outside its range.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DepthNoise {
    /// The standard deviation at the true depth `z`, in metres:
    /// `sigma[0] + sigma[1] z + sigma[2] z²`; a time-of-flight camera
    /// grows about linearly, a stereo camera with the square.
    pub sigma: [f64; 3],
    /// The share of pixels with a depth that read missing, in `[0, 1]`.
    pub dropout: f64,
    pub seed: u64,
}

impl CameraSpec {
    /// The intrinsics of the rendered image.
    pub fn info(&self) -> CameraInfo {
        let Intrinsics { fx, fy, cx, cy, distortion } = self.intrinsics;
        let (distortion_model, d) = distortion.camera_info();
        CameraInfo {
            width: self.width,
            height: self.height,
            k: [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0],
            distortion_model,
            d,
            ..Default::default()
        }
    }

    fn validate(&self) -> BulletResult<()> {
        let Intrinsics { fx, fy, cx, cy, distortion } = self.intrinsics;
        if self.width == 0 || self.height == 0 {
            return Err(invalid("a camera image needs pixels"));
        }
        if !(fx.is_finite() && fy.is_finite() && fx > 0.0 && fy > 0.0) {
            return Err(invalid("camera focal lengths must be finite and positive"));
        }
        if !(cx.is_finite() && cy.is_finite()) {
            return Err(invalid("a camera principal point must be finite"));
        }
        if !distortion.camera_info().1.iter().all(|d| d.is_finite()) {
            return Err(invalid("lens distortion coefficients must be finite"));
        }
        if !(self.near > 0.0 && self.far > self.near && self.far.is_finite()) {
            return Err(invalid(
                "a camera needs finite clipping distances with 0 < near < far",
            ));
        }
        if let Some(noise) = self.depth_noise {
            if !noise.sigma.iter().all(|s| s.is_finite() && *s >= 0.0) {
                return Err(invalid(
                    "depth noise coefficients must be finite and nonnegative",
                ));
            }
            if !(0.0..=1.0).contains(&noise.dropout) {
                return Err(invalid("a depth dropout share must be within [0, 1]"));
            }
        }
        Ok(())
    }
}

/// One rendered frame, row-major from the top-left pixel.
#[derive(Clone, Debug, PartialEq)]
pub struct RgbdFrame {
    pub width: u32,
    pub height: u32,
    /// Four bytes per pixel.
    pub rgba: Vec<u8>,
    /// Depth along the optical axis (m): the `z` of the point the pixel
    /// sees. `f32::INFINITY` marks a missing depth, the contract's one
    /// missing value, and nothing else is infinite: no surface before
    /// `far` (a surface at the far plane itself counts as none), and with
    /// [`DepthNoise`] a pixel dropped or read outside `near..=far`. No
    /// depth is ever a finite number beyond `far`.
    pub depth: Vec<f32>,
}

impl RgbdFrame {
    pub fn depth_at(&self, u: u32, v: u32) -> f32 {
        self.depth[(v * self.width + u) as usize]
    }
}

impl RsBullet {
    /// Renders `spec` now. The frame is stamped with simulation time
    /// ([`clock::SIMULATION`]); rendering does not step the world.
    pub fn render_rgbd(&mut self, spec: &CameraSpec) -> BulletResult<Stamped<RgbdFrame>> {
        let nanos = i64::try_from(self.simulated_time().as_nanos()).unwrap_or(i64::MAX);
        let frame = render(&mut self.client, spec, nanos)?;
        Ok(Stamped::new(
            Header::new(Stamp::new(clock::SIMULATION, nanos), spec.frame, 0),
            frame,
        ))
    }

    /// The `Camera` contract over this world, borrowed while in use.
    pub fn camera(&mut self, spec: CameraSpec) -> RsBulletCamera<'_> {
        RsBulletCamera { sim: self, spec, seq: 0 }
    }
}

/// A [`CameraSpec`] rendered from a borrowed world on each capture.
pub struct RsBulletCamera<'a> {
    sim: &'a mut RsBullet,
    spec: CameraSpec,
    seq: u64,
}

impl RsBulletCamera<'_> {
    /// Moves the camera, e.g. with the link it is mounted on.
    pub fn set_pose(&mut self, pose: na::Isometry3<f64>) {
        self.spec.pose = pose;
    }
}

impl Camera for RsBulletCamera<'_> {
    type Frame = RgbdFrame;

    fn camera_info(&self) -> CameraInfo {
        self.spec.info()
    }

    /// Renders the current state; the world is not stepped.
    fn capture(&mut self) -> RobotResult<Stamped<RgbdFrame>> {
        let mut frame = self
            .sim
            .render_rgbd(&self.spec)
            .map_err(|e| RobotException::CommandException(format!("camera: {e}")))?;
        self.seq += 1;
        frame.header.seq = self.seq;
        Ok(frame)
    }
}

fn invalid(message: &'static str) -> BulletError {
    BulletError::CommandFailed { message, code: -1 }
}

fn render(client: &mut PhysicsClient, spec: &CameraSpec, nanos: i64) -> BulletResult<RgbdFrame> {
    spec.validate()?;
    let Intrinsics { fx, fy, cx, cy, distortion } = spec.intrinsics;
    let mut frame = match distortion {
        Distortion::None => {
            let pinhole = Pinhole { width: spec.width, height: spec.height, fx, fy, cx, cy };
            render_pinhole(client, spec, &pinhole)?
        }
        Distortion::PlumbBob(_) | Distortion::RationalPolynomial(_) => {
            render_distorted(client, spec, &distortion.camera_info().1)?
        }
    };
    if let Some(noise) = &spec.depth_noise {
        noise.apply(&mut frame.depth, spec.near, spec.far, nanos);
    }
    Ok(frame)
}

/// An undistorted image to render.
struct Pinhole {
    width: u32,
    height: u32,
    fx: f64,
    fy: f64,
    cx: f64,
    cy: f64,
}

/// Renders an undistorted image with metric depth, missing depth infinite.
fn render_pinhole(
    client: &mut PhysicsClient,
    spec: &CameraSpec,
    lens: &Pinhole,
) -> BulletResult<RgbdFrame> {
    let eye = spec.pose.translation.vector;
    let forward = spec.pose.rotation * na::Vector3::z();
    // OpenGL's up is the image's -y.
    let up = spec.pose.rotation * -na::Vector3::y();
    let target = eye + forward;
    let f32s = |v: na::Vector3<f64>| [v.x as f32, v.y as f32, v.z as f32];
    let view = PhysicsClient::compute_view_matrix(f32s(eye), f32s(target), f32s(up));
    let (near, far) = (spec.near, spec.far);
    let (w, h) = (f64::from(lens.width), f64::from(lens.height));
    // Bullet's software renderer samples column u at u pixel widths right
    // of the frustum's left edge and, its rows flipped, row v at h - 1 - v
    // pixel heights above the bottom edge: not at pixel centres, which
    // would be half a pixel further in. The frustum is placed so that these
    // samples are the rays of the pixel centres, (u - cx) / fx and
    // (v - cy) / fy in normalised coordinates.
    let projection = PhysicsClient::compute_projection_matrix(
        (-near * lens.cx / lens.fx) as f32,
        (near * (w - lens.cx) / lens.fx) as f32,
        (near * (lens.cy + 1.0 - h) / lens.fy) as f32,
        (near * (lens.cy + 1.0) / lens.fy) as f32,
        near as f32,
        far as f32,
    );
    let options = CameraImageOptions {
        view_matrix: Some(view),
        projection_matrix: Some(projection),
        renderer: Some(Renderer::TinyRenderer),
        ..CameraImageOptions::new(lens.width as i32, lens.height as i32)
    };
    let image = client.get_camera_image(&options)?;
    // The depth buffer holds normalised device depth in [0, 1]. Where
    // nothing was hit it is the far plane, 1, which the software renderer
    // returns as a float32 a step or two below 1: anything that close to
    // the far plane is no hit, and so is a depth that rounds beyond it.
    let depth = image
        .depth
        .iter()
        .map(|&d| {
            let z = far * near / (far - (far - near) * f64::from(d));
            if f64::from(d) >= 1.0 - NO_HIT || z.is_nan() || z > far {
                f32::INFINITY
            } else {
                z as f32
            }
        })
        .collect();
    Ok(RgbdFrame {
        width: image.width as u32,
        height: image.height as u32,
        rgba: image.rgba,
        depth,
    })
}

/// Renders a distorted image: an undistorted one covering every pixel's
/// ray, finer by [`OVERSAMPLING`], resampled at the nearest pixel.
fn render_distorted(
    client: &mut PhysicsClient,
    spec: &CameraSpec,
    d: &[f64; 8],
) -> BulletResult<RgbdFrame> {
    let Intrinsics { fx, fy, cx, cy, .. } = spec.intrinsics;
    let pixels = spec.width as usize * spec.height as usize;
    let mut rays = Vec::with_capacity(pixels);
    let (mut x_min, mut x_max, mut y_min, mut y_max) = (
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::INFINITY,
        f64::NEG_INFINITY,
    );
    for v in 0..spec.height {
        for u in 0..spec.width {
            let seen = [(f64::from(u) - cx) / fx, (f64::from(v) - cy) / fy];
            let [x, y] = undistort(d, seen).ok_or_else(|| {
                invalid("the lens distortion cannot be inverted within the image")
            })?;
            x_min = x_min.min(x);
            x_max = x_max.max(x);
            y_min = y_min.min(y);
            y_max = y_max.max(y);
            rays.push([x, y]);
        }
    }
    // One rendered pixel of margin around the rays.
    let (rfx, rfy) = (OVERSAMPLING * fx, OVERSAMPLING * fy);
    let width = ((x_max - x_min) * rfx).ceil() + 3.0;
    let height = ((y_max - y_min) * rfy).ceil() + 3.0;
    if width > MAX_RENDER_SIDE || height > MAX_RENDER_SIDE {
        return Err(invalid(
            "the lens distortion spreads the image too far to render",
        ));
    }
    let lens = Pinhole {
        width: width as u32,
        height: height as u32,
        fx: rfx,
        fy: rfy,
        cx: 1.0 - x_min * rfx,
        cy: 1.0 - y_min * rfy,
    };
    let rendered = render_pinhole(client, spec, &lens)?;
    let mut rgba = Vec::with_capacity(pixels * 4);
    let mut depth = Vec::with_capacity(pixels);
    for [x, y] in rays {
        let u = (lens.fx * x + lens.cx).round() as usize;
        let v = (lens.fy * y + lens.cy).round() as usize;
        let at = v * rendered.width as usize + u;
        rgba.extend_from_slice(&rendered.rgba[4 * at..4 * at + 4]);
        depth.push(rendered.depth[at]);
    }
    Ok(RgbdFrame { width: spec.width, height: spec.height, rgba, depth })
}

/// Where the lens moves undistorted normalised coordinates `p`; see
/// [`Distortion`].
fn distort(d: &[f64; 8], [x, y]: [f64; 2]) -> [f64; 2] {
    let [k1, k2, p1, p2, k3, k4, k5, k6] = *d;
    let r2 = x * x + y * y;
    let s = (1.0 + r2 * (k1 + r2 * (k2 + r2 * k3))) / (1.0 + r2 * (k4 + r2 * (k5 + r2 * k6)));
    [
        x * s + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x),
        y * s + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y,
    ]
}

/// The undistorted normalised coordinates that the lens moves to `seen`,
/// by fixed-point iteration on [`distort`]: remove the tangential term
/// and divide by the radial factor at the current estimate. `None` when it
/// does not converge.
fn undistort(d: &[f64; 8], seen: [f64; 2]) -> Option<[f64; 2]> {
    let [k1, k2, p1, p2, k3, k4, k5, k6] = *d;
    let mut p = seen;
    for _ in 0..UNDISTORT_ITERATIONS {
        let [x, y] = p;
        let r2 = x * x + y * y;
        let s = (1.0 + r2 * (k1 + r2 * (k2 + r2 * k3))) / (1.0 + r2 * (k4 + r2 * (k5 + r2 * k6)));
        if !(s.is_finite() && s > 0.0) {
            return None;
        }
        p = [
            (seen[0] - 2.0 * p1 * x * y - p2 * (r2 + 2.0 * x * x)) / s,
            (seen[1] - p1 * (r2 + 2.0 * y * y) - 2.0 * p2 * x * y) / s,
        ];
        let back = distort(d, p);
        if (back[0] - seen[0]).abs() < UNDISTORT_RESIDUAL
            && (back[1] - seen[1]).abs() < UNDISTORT_RESIDUAL
        {
            return Some(p);
        }
    }
    None
}

impl DepthNoise {
    /// Adds the noise to the depths of a frame taken at `nanos`.
    fn apply(&self, depth: &mut [f32], near: f64, far: f64, nanos: i64) {
        let mut draws = SplitMix64(self.seed ^ SplitMix64(nanos as u64).next_u64());
        for depth in depth.iter_mut() {
            // Three draws for every pixel, so that a pixel's noise does not
            // depend on which other pixels have a depth.
            let (lost, a, b) = (draws.unit(), draws.unit(), draws.unit());
            if depth.is_infinite() {
                continue;
            }
            let z = f64::from(*depth);
            let sigma = self.sigma[0] + z * (self.sigma[1] + z * self.sigma[2]);
            let gaussian = (-2.0 * a.ln()).sqrt() * (TAU * b).cos();
            let read = z + sigma * gaussian;
            *depth = if lost < self.dropout || !(near..=far).contains(&read) {
                f32::INFINITY
            } else {
                read as f32
            };
        }
    }
}

/// SplitMix64 (Steele, Lea and Flood, 2014): enough for simulated noise,
/// and no dependency.
struct SplitMix64(u64);

impl SplitMix64 {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in `(0, 1]`.
    fn unit(&mut self) -> f64 {
        ((self.next_u64() >> 11) + 1) as f64 / (1u64 << 53) as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn undistortion_inverts_the_lens() {
        let d = [-0.28, 0.07, 1e-3, -5e-4, 0.0, 0.0, 0.0, 0.0];
        for seen in [[0.0, 0.0], [0.4, -0.3], [-0.55, 0.42], [0.1, 0.6]] {
            let p = undistort(&d, seen).unwrap();
            let back = distort(&d, p);
            assert!((back[0] - seen[0]).abs() < 1e-11 && (back[1] - seen[1]).abs() < 1e-11);
        }
        // A lens that folds back: s turns negative past r = 1.
        let folded = [-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        assert!(undistort(&folded, [2.0, 0.0]).is_none());
    }

    #[test]
    fn noise_draws_depend_on_seed_time_and_pixel_only() {
        let noise = DepthNoise { sigma: [0.01, 0.0, 0.0], dropout: 0.0, seed: 7 };
        let mut a = vec![1.0f32, f32::INFINITY, 1.0];
        let mut b = vec![1.0f32, 1.0, 1.0];
        noise.apply(&mut a, 0.1, 3.0, 5);
        noise.apply(&mut b, 0.1, 3.0, 5);
        assert!(a[1].is_infinite());
        assert_eq!((a[0], a[2]), (b[0], b[2]));
        let mut later = vec![1.0f32, 1.0, 1.0];
        noise.apply(&mut later, 0.1, 3.0, 6);
        assert_ne!(later, b);
    }
}
