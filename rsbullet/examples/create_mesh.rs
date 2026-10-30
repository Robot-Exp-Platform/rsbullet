use std::{f64::consts::FRAC_2_PI, thread::sleep, time::Duration};

use nalgebra as na;
use rsbullet::{
    BulletResult, CameraImageOptions, CollisionGeometry, CollisionId, CollisionShapeOptions,
    ControlMode, DebugVisualizerFlag, DebugVisualizerOptions, DynamicsUpdate, JointType, Mode,
    MultiBodyBase, MultiBodyCreateOptions, MultiBodyLink, PhysicsClient, Renderer,
    ResetDebugVisualizerCameraOptions,
};

const VERTICES: [[f64; 3]; 8] = [
    [-0.246350, -0.246483, -0.000624],
    [-0.151407, -0.176325, 0.172867],
    [-0.246350, 0.249205, -0.000624],
    [-0.151407, 0.129477, 0.172867],
    [0.249338, -0.246483, -0.000624],
    [0.154395, -0.176325, 0.172867],
    [0.249338, 0.249205, -0.000624],
    [0.154395, 0.129477, 0.172867],
];
const INDICES: [i32; 36] = [
    0, 3, 2, 3, 6, 2, 7, 4, 6, 5, 0, 4, 6, 0, 2, 3, 5, 7, 0, 1, 3, 3, 7, 6, 7, 5, 4, 5, 1, 0, 6, 4,
    0, 3, 1, 5,
];

fn main() -> BulletResult<()> {
    let mut client = PhysicsClient::connect(Mode::Gui)?;
    client
        .set_default_search_path()?
        .set_gravity([0., 0., -10.])?
        .reset_simulation()?;

    client
        .reset_debug_visualizer_camera(&ResetDebugVisualizerCameraOptions {
            distance: 15.,
            yaw: -346.,
            pitch: -16.,
            target: [-15., 0., 1.],
        })?
        .configure_debug_visualizer(&DebugVisualizerOptions::Flag(
            DebugVisualizerFlag::CovEnableWireframe,
            false,
        ))?;

    let col_sphere = client.create_collision_shape(
        &CollisionGeometry::Sphere { radius: 0.05 },
        None::<CollisionShapeOptions>,
    )?;
    let stone = client.create_collision_shape(
        &CollisionGeometry::Mesh { vertices: &VERTICES, indices: &INDICES, scale: [0.; 3] },
        None::<CollisionShapeOptions>,
    )?;
    let col_box = client.create_collision_shape(
        &CollisionGeometry::Box { half_extents: [0.5, 2.5, 00.1] },
        None::<CollisionShapeOptions>,
    )?;

    let mut segment_start = 0.;

    for _ in 0..5 {
        client.create_multi_body(&MultiBodyCreateOptions {
            base: MultiBodyBase {
                mass: 0.,
                collision_shape: CollisionId(col_box),
                pose: na::Isometry3::translation(segment_start, 0., -0.1),
                ..Default::default()
            },
            ..Default::default()
        })?;
        segment_start -= 1.;
    }

    for i in 0..5 {
        client.create_multi_body(&MultiBodyCreateOptions {
            base: MultiBodyBase {
                mass: 0.,
                collision_shape: CollisionId(col_box),
                pose: na::Isometry3::translation(segment_start, 0., -0.1 + 1. * (i % 2) as f64),
                ..Default::default()
            },
            ..Default::default()
        })?;
        segment_start -= 1.;
    }

    for i in 0..5 {
        client.create_multi_body(&MultiBodyCreateOptions {
            base: MultiBodyBase {
                mass: 0.,
                collision_shape: CollisionId(col_box),
                pose: na::Isometry3::translation(segment_start, 0., -0.1),
                ..Default::default()
            },
            ..Default::default()
        })?;
        segment_start -= 1.;
        if i % 2 == 0 {
            let transform = [segment_start, (i % 3) as f64, -0.1 + 2.5].into();
            let rotation = na::UnitQuaternion::from_euler_angles(FRAC_2_PI, 0., FRAC_2_PI);
            client.create_multi_body(&MultiBodyCreateOptions {
                base: MultiBodyBase {
                    mass: 0.,
                    collision_shape: CollisionId(col_box),
                    pose: na::Isometry3::from_parts(transform, rotation),
                    ..Default::default()
                },
                ..Default::default()
            })?;
        }
    }

    for i in 0..5 {
        client.create_multi_body(&MultiBodyCreateOptions {
            base: MultiBodyBase {
                mass: 0.,
                collision_shape: CollisionId(col_box),
                pose: na::Isometry3::translation(segment_start, 0., -0.1),
                ..Default::default()
            },
            ..Default::default()
        })?;
        for j in 0..4 {
            let pose = [segment_start, (i % 2) as f64 / 2. + j as f64 - 2.0, 0.].into();
            client.create_multi_body(&MultiBodyCreateOptions {
                base: MultiBodyBase {
                    mass: 0.,
                    collision_shape: CollisionId(stone),
                    pose,
                    ..Default::default()
                },
                ..Default::default()
            })?;
        }
        segment_start -= 1.;
    }

    let base_pose = [segment_start, 0., -0.1].into();

    for _ in 0..5 {
        let box_id = client.create_multi_body(&MultiBodyCreateOptions {
            base: MultiBodyBase {
                mass: 0.,
                pose: base_pose,
                collision_shape: CollisionId(col_sphere),
                ..Default::default()
            },
            links: &[MultiBodyLink {
                mass: 1.,
                joint_type: JointType::Revolute,
                collision_shape: CollisionId(col_box),
                parent_index: Some(0),
                joint_axis: [1., 0., 0.],
                ..Default::default()
            }],
            ..Default::default()
        })?;
        client.change_dynamics(
            box_id,
            -1,
            &DynamicsUpdate {
                spinning_friction: Some(0.001),
                rolling_friction: Some(0.001),
                linear_damping: Some(0.),
                ..Default::default()
            },
        )?;
        for joint in 0..client.get_num_joints(box_id) {
            let target_velocity = if joint % 2 == 0 { 10. } else { -10. };
            client.set_joint_motor_control(
                box_id,
                joint,
                ControlMode::Velocity(target_velocity),
                None,
            )?;
        }
    }

    loop {
        let cam = client.get_debug_visualizer_camera()?;
        client.get_camera_image(&CameraImageOptions {
            width: 256,
            height: 256,
            view_matrix: Some(cam.view_matrix),
            projection_matrix: Some(cam.projection_matrix),
            renderer: Some(Renderer::BulletHardwareOpenGl),
            ..Default::default()
        })?;
        client.step_simulation()?;
        sleep(Duration::from_secs_f64(0.01));
    }
}
