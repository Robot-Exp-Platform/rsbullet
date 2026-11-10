//! The base's URDF frame against its centre of mass, on a base whose
//! inertial frame is offset and turned. DIRECT engine, no graphics.
use std::time::Duration;

use nalgebra::{Isometry3, Translation3, UnitQuaternion, Vector3};
use rsbullet_core::{LoadModelFlags, Mode, PhysicsClient, UrdfOptions};

const URDF: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/assets/offset_inertia_base.urdf"
);

/// Pose tolerance (m, rad): Bullet holds poses in single precision.
const TOL: f64 = 1e-6;

/// The base link's `<inertial><origin>` in the URDF.
fn urdf_inertial() -> Isometry3<f64> {
    Isometry3::from_parts(
        Translation3::new(0.12, -0.05, 0.03),
        UnitQuaternion::from_euler_angles(0.0, 0.0, 0.3),
    )
}

/// Where the tests load the base frame: off the origin and turned about
/// all three axes.
fn load_pose() -> Isometry3<f64> {
    Isometry3::from_parts(
        Translation3::new(0.4, -0.7, 1.2),
        UnitQuaternion::from_euler_angles(0.2, -0.1, 0.8),
    )
}

fn client() -> Result<PhysicsClient, Box<dyn std::error::Error>> {
    let mut client = PhysicsClient::connect(Mode::Direct)?;
    client
        .set_time_step(Duration::from_millis(1))?
        .set_gravity([0.0, 0.0, 0.0])?;
    Ok(client)
}

fn load(
    client: &mut PhysicsClient,
    options: UrdfOptions,
) -> Result<i32, Box<dyn std::error::Error>> {
    let base = Some(load_pose());
    Ok(client.load_urdf(URDF, Some(UrdfOptions { base, ..options }))?)
}

fn assert_pose_near(what: &str, actual: Isometry3<f64>, expected: Isometry3<f64>, tol: f64) {
    let offset = (actual.translation.vector - expected.translation.vector).norm();
    let angle = actual.rotation.angle_to(&expected.rotation);
    assert!(
        offset < tol && angle < tol,
        "{what}: {offset:.3e} m and {angle:.3e} rad from the expected pose"
    );
}

#[test]
fn the_base_frame_is_where_the_urdf_is_loaded_and_the_centre_of_mass_is_not()
-> Result<(), Box<dyn std::error::Error>> {
    let mut client = client()?;
    let body = load(&mut client, UrdfOptions::default())?;

    // Bullet's own base pose is the centre of mass: the URDF's inertial
    // origin away from where the frame was loaded.
    let com = client.get_base_position_and_orientation(body)?;
    assert_pose_near("centre of mass", com, load_pose() * urdf_inertial(), TOL);
    let apart = (com.translation.vector - load_pose().translation.vector).norm();
    assert!(
        apart > 0.1,
        "the centre of mass is {apart} m from the frame"
    );

    let frame = client.get_base_frame_pose(body)?;
    assert_pose_near("base frame", frame, load_pose(), TOL);

    // The hinge sits 0.3 m ahead of the base frame, so the arm's link frame
    // at zero angle is the base frame moved by the joint origin.
    let arm = client
        .get_link_state(body, 0, true, false)?
        .world_link_frame;
    assert_pose_near(
        "arm link frame",
        arm,
        frame * Isometry3::translation(0.3, 0.0, 0.0),
        TOL,
    );

    // The same frame as Bullet's dynamics info gives it.
    let info = client.get_dynamics_info(body, -1)?;
    let [x, y, z] = info.local_inertia_position;
    let [qx, qy, qz, qw] = info.local_inertia_orientation;
    let inertial = Isometry3::from_parts(
        Translation3::new(x, y, z),
        UnitQuaternion::from_quaternion(nalgebra::Quaternion::new(qw, qx, qy, qz)),
    );
    assert_pose_near("dynamics info", inertial, urdf_inertial(), TOL);
    Ok(())
}

#[test]
fn the_base_frame_holds_with_the_file_inertia_and_maximal_coordinates()
-> Result<(), Box<dyn std::error::Error>> {
    let mut client = client()?;
    // With the file's (non-diagonal) inertia Bullet turns the inertial frame
    // to the principal axes, so the centre-of-mass orientation is neither
    // the frame's nor the URDF inertial origin's.
    let body = load(
        &mut client,
        UrdfOptions {
            flags: Some(LoadModelFlags::URDF_USE_INERTIA_FROM_FILE),
            ..Default::default()
        },
    )?;
    assert_pose_near(
        "base frame, file inertia",
        client.get_base_frame_pose(body)?,
        load_pose(),
        TOL,
    );
    let com = client.get_base_position_and_orientation(body)?;
    let turned = com
        .rotation
        .angle_to(&(load_pose() * urdf_inertial()).rotation);
    assert!(turned > 0.05, "principal axes turned by {turned} rad");

    let rigid = load(
        &mut client,
        UrdfOptions { use_maximal_coordinates: Some(true), ..Default::default() },
    )?;
    assert_pose_near(
        "base frame, maximal coordinates",
        client.get_base_frame_pose(rigid)?,
        load_pose(),
        TOL,
    );
    Ok(())
}

#[test]
fn resetting_the_base_frame_places_the_frame() -> Result<(), Box<dyn std::error::Error>> {
    let mut client = client()?;
    let body = load(&mut client, UrdfOptions::default())?;
    let target = Isometry3::from_parts(
        Translation3::new(-1.0, 2.0, 0.5),
        UnitQuaternion::from_euler_angles(-0.4, 0.3, -2.0),
    );
    client.reset_base_frame_pose(body, target)?;
    assert_pose_near("base frame", client.get_base_frame_pose(body)?, target, TOL);
    assert_pose_near(
        "centre of mass",
        client.get_base_position_and_orientation(body)?,
        target * urdf_inertial(),
        TOL,
    );

    // The pybullet reset places the centre of mass, not the frame.
    let (p, q) = (target.translation.vector, target.rotation);
    client.reset_base_position_and_orientation(body, [p.x, p.y, p.z], [q.i, q.j, q.k, q.w])?;
    assert_pose_near(
        "centre of mass after the centre-of-mass reset",
        client.get_base_position_and_orientation(body)?,
        target,
        TOL,
    );
    assert_pose_near(
        "base frame after the centre-of-mass reset",
        client.get_base_frame_pose(body)?,
        target * urdf_inertial().inverse(),
        TOL,
    );
    Ok(())
}

#[test]
fn the_base_frame_velocity_is_the_frame_origins() -> Result<(), Box<dyn std::error::Error>> {
    let mut client = client()?;
    let body = load(&mut client, UrdfOptions::default())?;
    let dt = 1e-3;
    // Spin the base about the vertical through its centre of mass, which
    // stays put while the frame origin, 0.13 m away, circles it.
    client.reset_base_velocity(body, Some([0.0; 3]), Some([0.0, 0.0, 2.0]))?;
    let com_velocity = client.get_base_velocity(body)?;
    let before = client.get_base_frame_velocity(body)?;
    assert!(
        Vector3::new(com_velocity[0], com_velocity[1], com_velocity[2]).norm() < TOL,
        "the centre of mass moves at {com_velocity:?}"
    );
    assert_eq!(
        before[3..],
        com_velocity[3..],
        "the angular velocity is the body's"
    );

    let start = client.get_base_frame_pose(body)?.translation.vector;
    client.step_simulation()?;
    let end = client.get_base_frame_pose(body)?.translation.vector;
    let after = client.get_base_frame_velocity(body)?;

    let travelled = (end - start) / dt;
    let reported = (Vector3::new(before[0], before[1], before[2])
        + Vector3::new(after[0], after[1], after[2]))
        / 2.0;
    assert!(
        reported.norm() > 0.2,
        "the frame origin moves at {} m/s",
        reported.norm()
    );
    assert!(
        (travelled - reported).norm() < 2e-3,
        "the frame origin travelled at {travelled:?} m/s, reported {reported:?}"
    );
    Ok(())
}
