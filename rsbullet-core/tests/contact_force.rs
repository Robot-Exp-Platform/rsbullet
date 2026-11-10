//! Contact forces with and without integration sub-steps, against the
//! weight of a box at rest. DIRECT engine, no graphics.
use std::time::Duration;

use nalgebra::{Isometry3, Translation3, UnitQuaternion, Vector3};
use rsbullet_core::{
    CollisionGeometry, CollisionId, CollisionShapeOptions, Mode, MultiBodyBase,
    MultiBodyCreateOptions, PhysicsClient, PhysicsEngineParametersUpdate,
};

const MASS: f64 = 2.0;
const G: f64 = 9.81;

/// The summed normal force and the magnitude of the summed friction force
/// (N) on a 2 kg box resting for one second on a floor tilted by `incline`
/// (rad), stepped at 240 Hz in `sub_steps` sub-steps (0: Bullet's default,
/// none).
fn resting_box(sub_steps: usize, incline: f64) -> Result<(f64, f64), Box<dyn std::error::Error>> {
    let mut client = PhysicsClient::connect(Mode::Direct)?;
    client
        .set_time_step(Duration::from_secs_f64(1.0 / 240.0))?
        .set_gravity([0.0, 0.0, -G])?;
    if sub_steps > 0 {
        client.set_physics_engine_parameter(&PhysicsEngineParametersUpdate {
            num_sub_steps: Some(sub_steps),
            ..Default::default()
        })?;
    }
    let tilt = UnitQuaternion::from_euler_angles(0.0, incline, 0.0);
    let body = |client: &mut PhysicsClient, half: [f64; 3], mass: f64, centre: f64| {
        let shape = client.create_collision_shape(
            &CollisionGeometry::Box { half_extents: half },
            Some(CollisionShapeOptions::default()),
        )?;
        let at = tilt * Vector3::new(0.0, 0.0, centre);
        client.create_multi_body(&MultiBodyCreateOptions {
            base: MultiBodyBase {
                mass,
                pose: Isometry3::from_parts(Translation3::from(at), tilt),
                collision_shape: CollisionId(shape),
                ..Default::default()
            },
            ..Default::default()
        })
    };
    let floor = body(&mut client, [1.0, 1.0, 0.05], 0.0, -0.05)?;
    let load = body(&mut client, [0.1, 0.1, 0.05], MASS, 0.0505)?;
    for _ in 0..240 {
        client.step_simulation()?;
    }
    let contacts = client.get_contact_points(Some(load), Some(floor), None, None)?;
    assert!(!contacts.is_empty(), "the box rests on the floor");
    let normal: f64 = contacts.iter().map(|c| c.normal_force).sum();
    let friction: Vector3<f64> = contacts
        .iter()
        .map(|c| {
            Vector3::from(c.linear_friction_direction_1) * c.linear_friction_force_1
                + Vector3::from(c.linear_friction_direction_2) * c.linear_friction_force_2
        })
        .sum();
    Ok((normal, friction.norm()))
}

fn assert_near(what: &str, actual: f64, expected: f64, relative: f64) {
    assert!(
        (actual - expected).abs() <= relative * expected,
        "{what}: {actual:.4} N, expected {expected:.4} N within {:.1} %",
        relative * 100.0
    );
}

#[test]
fn the_support_force_is_the_weight_with_and_without_sub_steps()
-> Result<(), Box<dyn std::error::Error>> {
    let weight = MASS * G;
    let (without, _) = resting_box(0, 0.0)?;
    assert_near("no sub-steps", without, weight, 0.01);
    for sub_steps in [1, 2, 4, 8] {
        let (with, _) = resting_box(sub_steps, 0.0)?;
        assert_near(&format!("{sub_steps} sub-steps"), with, weight, 0.01);
        assert_near(
            &format!("{sub_steps} sub-steps against none"),
            with,
            without,
            0.01,
        );
    }
    Ok(())
}

#[test]
fn friction_holds_the_box_on_a_slope_with_and_without_sub_steps()
-> Result<(), Box<dyn std::error::Error>> {
    let incline: f64 = 0.2;
    for sub_steps in [0, 4] {
        let (normal, friction) = resting_box(sub_steps, incline)?;
        assert_near(
            &format!("normal force, {sub_steps} sub-steps"),
            normal,
            MASS * G * incline.cos(),
            0.02,
        );
        assert_near(
            &format!("friction force, {sub_steps} sub-steps"),
            friction,
            MASS * G * incline.sin(),
            0.02,
        );
    }
    Ok(())
}
