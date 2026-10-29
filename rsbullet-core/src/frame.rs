//! Bullet's raw frame layout: a position (XYZ) followed by a quaternion (XYZW).

use nalgebra as na;

/// An isometry from a position and an XYZW quaternion.
fn isometry_from_raw_parts(position: [f64; 3], orientation_xyzw: [f64; 4]) -> na::Isometry3<f64> {
    let translation = na::Translation3::new(position[0], position[1], position[2]);
    let quaternion = na::Quaternion::new(
        orientation_xyzw[3],
        orientation_xyzw[0],
        orientation_xyzw[1],
        orientation_xyzw[2],
    );
    na::Isometry3::from_parts(translation, na::UnitQuaternion::from_quaternion(quaternion))
}

/// An isometry from a `[position, orientation]` frame.
pub(crate) fn isometry_from_frame(frame: &[f64; 7]) -> na::Isometry3<f64> {
    isometry_from_raw_parts(
        [frame[0], frame[1], frame[2]],
        [frame[3], frame[4], frame[5], frame[6]],
    )
}

/// The position (XYZ) and quaternion (XYZW) of an isometry.
pub(crate) fn isometry_to_raw_parts(transform: &na::Isometry3<f64>) -> ([f64; 3], [f64; 4]) {
    let rotation = transform.rotation;
    (
        transform.translation.into(),
        [rotation.i, rotation.j, rotation.k, rotation.w],
    )
}

/// Writes an isometry into a `[position, orientation]` frame.
pub(crate) fn isometry_write_to_frame(transform: &na::Isometry3<f64>, frame: &mut [f64; 7]) {
    let (position, orientation) = isometry_to_raw_parts(transform);
    frame[..3].copy_from_slice(&position);
    frame[3..7].copy_from_slice(&orientation);
}
