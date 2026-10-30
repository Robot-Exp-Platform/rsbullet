//! A simulated RGB-D camera behind the shared `Camera` contract.

use nalgebra as na;
use roplat_behavior::{
    Camera, RobotException, RobotResult,
    msgs::{CameraInfo, FrameId, Header, Stamp, Stamped, clock},
};
use rsbullet_core::{BulletResult, CameraImageOptions, PhysicsClient, Renderer};

use crate::RsBullet;

/// Normalised depth this close to 1 is the far plane: no hit.
const NO_HIT: f64 = 1e-6;

/// Where a simulated camera is and what it sees: a pinhole camera rendered
/// by Bullet's software renderer, which also works without a display
/// ([`Mode::Direct`](crate::Mode::Direct)).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CameraSpec {
    pub width: u32,
    pub height: u32,
    /// Vertical field of view (rad).
    pub fov_y: f64,
    /// Nearest and farthest distances rendered (m).
    pub near: f64,
    pub far: f64,
    /// The optical frame in the world: `z` along the view, `x` right and
    /// `y` down in the image, as `CameraInfo` and depth back-projection
    /// expect.
    pub pose: na::Isometry3<f64>,
    /// The optical frame named in each frame's header.
    pub frame: FrameId,
}

impl CameraSpec {
    /// The intrinsics of the rendered image: square pixels, principal point
    /// at the image centre, no distortion.
    pub fn info(&self) -> CameraInfo {
        let f = f64::from(self.height) / 2.0 / (self.fov_y / 2.0).tan();
        let (cx, cy) = (
            (f64::from(self.width) - 1.0) / 2.0,
            (f64::from(self.height) - 1.0) / 2.0,
        );
        CameraInfo {
            width: self.width,
            height: self.height,
            k: [f, 0.0, cx, 0.0, f, cy, 0.0, 0.0, 1.0],
            ..Default::default()
        }
    }
}

/// One rendered frame, row-major from the top-left pixel.
#[derive(Clone, Debug, PartialEq)]
pub struct RgbdFrame {
    pub width: u32,
    pub height: u32,
    /// Four bytes per pixel.
    pub rgba: Vec<u8>,
    /// Distance along the view axis (m); infinite where nothing was hit
    /// before `far` (a surface at the far plane itself counts as no hit).
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
        let frame = render(&mut self.client, spec)?;
        let nanos = i64::try_from(self.simulated_time().as_nanos()).unwrap_or(i64::MAX);
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

fn render(client: &mut PhysicsClient, spec: &CameraSpec) -> BulletResult<RgbdFrame> {
    let eye = spec.pose.translation.vector;
    let forward = spec.pose.rotation * na::Vector3::z();
    // OpenGL's up is the image's -y.
    let up = spec.pose.rotation * -na::Vector3::y();
    let target = eye + forward;
    let f32s = |v: na::Vector3<f64>| [v.x as f32, v.y as f32, v.z as f32];
    let view = PhysicsClient::compute_view_matrix(f32s(eye), f32s(target), f32s(up));
    let projection = PhysicsClient::compute_projection_matrix_fov(
        spec.fov_y.to_degrees() as f32,
        spec.width as f32 / spec.height as f32,
        spec.near as f32,
        spec.far as f32,
    );
    let options = CameraImageOptions {
        view_matrix: Some(view),
        projection_matrix: Some(projection),
        renderer: Some(Renderer::TinyRenderer),
        ..CameraImageOptions::new(spec.width as i32, spec.height as i32)
    };
    let image = client.get_camera_image(&options)?;
    let (near, far) = (spec.near, spec.far);
    // The depth buffer holds normalised device depth in [0, 1]. Where
    // nothing was hit it is the far plane, 1, which the software renderer
    // returns as a float32 a step or two below 1: anything that close to
    // the far plane is no hit.
    let depth = image
        .depth
        .iter()
        .map(|&d| {
            if f64::from(d) >= 1.0 - NO_HIT {
                f32::INFINITY
            } else {
                (far * near / (far - (far - near) * f64::from(d))) as f32
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
