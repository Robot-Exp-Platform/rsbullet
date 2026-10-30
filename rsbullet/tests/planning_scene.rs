//! `roplat_scene` and Bullet measure the same Panda.
//!
//! Both read Bullet's `franka_panda/panda.urdf` and make each OBJ collision
//! mesh convex; Bullet's URDF importer adds a 1 mm margin to each hull. In
//! random arm configurations Bullet must never find a pair the scene checks
//! closer than the scene does, less the two margins, so the scene never calls
//! free what Bullet finds in contact. Bullet often reports more, up to about
//! 10 mm in this sample; the scene's own tests check its distances against an
//! exact reference.

use std::{collections::BTreeMap, path::PathBuf};

use roplat_description::RobotModel;
use roplat_scene::PlanningScene;
use rsbullet::{ClosestPointsOptions, Mode, RsBullet, UrdfOptions};

const PANDA: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../rsbullet-sys/bullet3/examples/pybullet/gym/pybullet_data/franka_panda"
);
const ARM: [&str; 7] = [
    "panda_joint1",
    "panda_joint2",
    "panda_joint3",
    "panda_joint4",
    "panda_joint5",
    "panda_joint6",
    "panda_joint7",
];
const FINGERS: [&str; 2] = ["panda_finger_joint1", "panda_finger_joint2"];
/// Pairs closer than this (m) are compared.
const RANGE: f64 = 0.05;
/// Bullet's margin on each hull.
const MARGIN: f64 = 0.001;

#[test]
fn bullet_never_finds_links_closer_than_the_scene() {
    let dir = PathBuf::from(PANDA);
    let model = RobotModel::from_urdf_file(dir.join("panda.urdf")).unwrap();
    let mut scene = PlanningScene::new(model.clone(), ARM, |name| {
        dir.join(name.trim_start_matches("package://"))
    })
    .unwrap();
    scene.set_padding(RANGE).unwrap();

    let mut world = RsBullet::new(Mode::Direct).unwrap();
    let client = &mut world.client;
    let panda = client
        .load_urdf(
            dir.join("panda.urdf"),
            Some(UrdfOptions { use_fixed_base: true, ..Default::default() }),
        )
        .unwrap();
    // Bullet's link index of each link name, and joint index of each joint.
    let mut links = BTreeMap::from([(model.root().name.clone(), -1)]);
    let mut joints = BTreeMap::new();
    for index in 0..client.get_num_joints(panda) {
        let info = client.get_joint_info(panda, index).unwrap();
        links.insert(info.link_name, index);
        joints.insert(info.joint_name, index);
    }
    let finger = 0.04;
    for name in FINGERS {
        scene.set_joint(name, finger).unwrap();
        client
            .reset_joint_state(panda, joints[name], finger, None)
            .unwrap();
    }

    // The pairs the scene checks: links with geometry, not allowed to touch.
    let solid: Vec<&str> = model
        .links
        .iter()
        .filter(|link| !link.collisions.is_empty())
        .map(|link| link.name.as_str())
        .collect();
    let pairs: Vec<(&str, &str)> = solid
        .iter()
        .enumerate()
        .flat_map(|(i, &a)| solid[i + 1..].iter().map(move |&b| (a, b)))
        .filter(|(a, b)| !scene.allowed().is_allowed(a, b))
        .collect();
    assert_eq!(pairs.len(), 45, "{pairs:?}");

    let (lower, upper): (Vec<f64>, Vec<f64>) = ARM
        .iter()
        .map(|name| {
            let limit = model.joint(name).unwrap().limit.unwrap();
            (limit.lower, limit.upper)
        })
        .unzip();
    let mut seed = 20261028u64;
    let mut uniform = || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        (seed >> 11) as f64 / (1u64 << 53) as f64
    };

    let (mut compared, mut touching, mut agreeing) = (0, 0, 0);
    for _ in 0..300 {
        let q: [f64; 7] = std::array::from_fn(|i| lower[i] + (upper[i] - lower[i]) * uniform());
        for (name, position) in ARM.iter().zip(q) {
            client
                .reset_joint_state(panda, joints[*name], position, None)
                .unwrap();
        }
        let near: BTreeMap<(String, String), f64> = scene
            .contacts(&q)
            .into_iter()
            .map(|contact| ((contact.first, contact.second), contact.distance))
            .collect();
        for &(a, b) in &pairs {
            let options = ClosestPointsOptions {
                link_index_a: Some(links[a]),
                link_index_b: Some(links[b]),
                ..Default::default()
            };
            let bullet = client
                .get_closest_points(panda, panda, RANGE, &options)
                .unwrap()
                .iter()
                .map(|point| point.contact_distance)
                .reduce(f64::min);
            let ours = near.get(&(a.to_owned(), b.to_owned())).copied();
            if let Some(ours) = ours {
                compared += 1;
                touching += usize::from(ours <= 0.0);
                agreeing += usize::from(
                    bullet.is_some_and(|bullet| (bullet + 2.0 * MARGIN - ours).abs() < 5e-4),
                );
            }
            if let Some(bullet) = bullet {
                // Beyond the range when the scene does not report the pair.
                let floor = ours.unwrap_or(RANGE) - 2.0 * MARGIN - 1e-4;
                assert!(
                    bullet >= floor,
                    "{a}–{b} at {q:?}: scene {ours:?}, Bullet {bullet:.5} m"
                );
            }
        }
    }
    println!(
        "{compared} pairs within range, {touching} touching, {agreeing} within 0.5 mm of Bullet"
    );
    assert!(
        compared > 300 && touching > 30,
        "{compared} compared, {touching} touching"
    );
}
