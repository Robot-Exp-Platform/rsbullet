//! An entity loaded through `roplat_world::EntityBuilder` is a static body
//! at the base it was given, its shape at the body's origin, and `load`
//! returns that body, the id the segmentation mask reports for its pixels.
//!
//! The base used to be applied twice, as the shape's offset in the body and
//! as the body's pose, which put the shape at `base * base`; and `load`
//! returned only the shape, which no mask or body query takes.

use nalgebra::{Isometry3, Translation3, UnitQuaternion, Vector3};
use roplat_world::{AddCollision, AddVisual, Entity, EntityBuilder};
use rsbullet::{CameraImageOptions, Mode, PhysicsClient, RsBullet};

/// A base off the origin and turned about z, so that `base * base` lies
/// 0.36 m from it.
fn base() -> Isometry3<f64> {
    Isometry3::from_parts(
        Translation3::new(0.3, -0.2, 0.1),
        UnitQuaternion::from_axis_angle(&Vector3::z_axis(), 0.5),
    )
}

#[test]
fn a_collision_entity_sits_at_its_base() {
    let mut sim = RsBullet::new(Mode::Direct).unwrap();
    let loaded = sim
        .collision(Entity::Box { half_extents: [0.2, 0.1, 0.05] })
        .base(base())
        .load()
        .unwrap();
    // Bullet keeps a body's pose in single precision.
    let pose = sim.client.get_base_frame_pose(loaded.body).unwrap();
    assert!(
        (pose.translation.vector - base().translation.vector).norm() < 1e-6
            && pose.rotation.angle_to(&base().rotation) < 1e-6,
        "body at {pose:?}"
    );
    // The box's bounds, grown by Bullet's margin on every side alike, are
    // centred on the base.
    let aabb = sim.client.get_aabb(loaded.body, -1).unwrap();
    let centre = Vector3::from_fn(|i, _| (aabb.min[i] + aabb.max[i]) / 2.0);
    assert!(
        (centre - base().translation.vector).norm() < 1e-6,
        "shape centred at {centre:?}, base at {:?}",
        base().translation.vector
    );
}

#[test]
fn a_visual_entity_is_masked_with_the_body_load_returns() {
    let mut sim = RsBullet::new(Mode::Direct).unwrap();
    // A first entity, out of view, so that the box's body and shape ids
    // differ (body 1, visual shape 0).
    let away = sim
        .collision(Entity::Sphere { radius: 0.05 })
        .base(Isometry3::translation(5.0, 5.0, 5.0))
        .load()
        .unwrap();
    let boxed = sim
        .visual(Entity::Box { half_extents: [0.1, 0.1, 0.1] })
        .base(base())
        .load()
        .unwrap();
    assert_ne!(boxed.body, away.body);
    assert_ne!(boxed.body, boxed.shape.0);

    // Looking straight down at the base from 1.9 m above it.
    let at = base().translation.vector.cast::<f32>();
    let view = PhysicsClient::compute_view_matrix(
        [at.x, at.y, at.z + 1.9],
        [at.x, at.y, at.z],
        [0.0, 1.0, 0.0],
    );
    let projection = PhysicsClient::compute_projection_matrix_fov(40.0, 1.0, 0.1, 5.0);
    const SIDE: usize = 32;
    let image = sim
        .client
        .get_camera_image(&CameraImageOptions {
            width: SIDE as i32,
            height: SIDE as i32,
            view_matrix: Some(view),
            projection_matrix: Some(projection),
            ..Default::default()
        })
        .unwrap();
    let mask = &image.segmentation_mask;
    assert_eq!(mask.len(), SIDE * SIDE);
    let centre = mask[SIDE / 2 * SIDE + SIDE / 2];
    assert_eq!(centre, boxed.body, "the pixel under the base");
    assert!(
        mask.iter().all(|&id| id == boxed.body || id == -1),
        "{mask:?}"
    );
}
