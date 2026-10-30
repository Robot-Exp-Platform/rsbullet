use nalgebra as na;
use rsbullet::*;
use std::time::Duration;

fn main() -> BulletResult<()> {
    let mut client = PhysicsClient::connect(Mode::Gui)?;
    client
        .set_default_search_path()?
        .set_physics_engine_parameter(&PhysicsEngineParametersUpdate {
            num_solver_iterations: Some(10),
            ..Default::default()
        })?
        .set_time_step(Duration::from_secs_f64(1. / 120.))?;

    let log_id = client.start_state_logging(
        LoggingType::ProfileTimings,
        "visualShapeBench.json",
        None::<()>,
    )?;

    let plane = client.load_urdf(
        "plane_transparent.urdf",
        Some(UrdfOptions { use_maximal_coordinates: Some(true), ..Default::default() }),
    )?;

    client
        .configure_debug_visualizer(&DebugVisualizerOptions::Flag(
            DebugVisualizerFlag::CovEnableRendering,
            false,
        ))?
        .configure_debug_visualizer(&DebugVisualizerOptions::Flag(
            DebugVisualizerFlag::CovEnablePlanarReflection,
            true,
        ))?
        .configure_debug_visualizer(&DebugVisualizerOptions::Flag(
            DebugVisualizerFlag::CovEnableGui,
            false,
        ))?
        .configure_debug_visualizer(&DebugVisualizerOptions::Flag(
            DebugVisualizerFlag::CovEnableTinyRenderer,
            false,
        ))?;

    let shift = [0., -0.02, 0.];
    let mesh_scale = [0.1, 0.1, 0.1];

    let visual_shape = client.create_visual_shape(
        &VisualGeometry::Mesh { file: "duck.obj", scale: mesh_scale },
        Some(VisualShapeOptions {
            transform: shift.into(),
            rgba: [1., 1., 1., 1.],
            specular: [0.4, 0.4, 0.],
            flags: None,
        }),
    )?;
    let collision_shape = client.create_collision_shape(
        &CollisionGeometry::MeshFile { file: "duck_vhacd.obj", scale: mesh_scale },
        Some(na::Isometry3::from(shift)),
    )?;

    let range_x = 3;
    let range_y = 3;

    for i in 0..range_x {
        for j in 0..range_y {
            client.create_multi_body(&MultiBodyCreateOptions {
                base: MultiBodyBase {
                    mass: 1.,
                    collision_shape: CollisionId(collision_shape),
                    visual_shape: VisualId(visual_shape),
                    pose: na::Isometry3::translation(
                        ((-range_x as f64 / 2.) + i as f64) * mesh_scale[0] * 2.,
                        ((-range_y as f64 / 2.) + j as f64) * mesh_scale[1] * 2.,
                        1.,
                    ),
                    inertial_pose: na::Isometry3::identity(),
                },
                use_maximal_coordinates: true,
                ..Default::default()
            })?;
        }
    }

    client
        .configure_debug_visualizer(&DebugVisualizerOptions::Flag(
            DebugVisualizerFlag::CovEnableRendering,
            true,
        ))?
        .stop_state_logging(log_id)?
        .set_gravity([0., 0., -9.81])?
        .set_real_time_simulation(true)?;

    let colors = [
        [1., 0., 0., 1.],
        [0., 1., 0., 1.],
        [0., 0., 1., 1.],
        [1., 1., 1., 1.],
    ];
    let mut current_color = 0;

    loop {
        let mouse_events = client.get_mouse_events()?;
        for event in mouse_events {
            match event {
                MouseEvent::Move { .. } => {}
                MouseEvent::Button { mouse_x, mouse_y, button_index, button_state } => {
                    if button_state.was_triggered() && button_index == 0 {
                        eprintln!("Mouse click at ({}, {})", mouse_x, mouse_y);
                        let (ray_from, ray_to) = get_ray_from_to(
                            client.get_debug_visualizer_camera()?,
                            mouse_x,
                            mouse_y,
                        );
                        eprintln!("Ray from: {:?}, Ray to: {:?}", ray_from, ray_to);
                        let ray_info = client.ray_test(ray_from, ray_to, None, None)?;
                        eprintln!("{:?}", ray_info);
                        for ray in ray_info {
                            if ray.object_unique_id != plane {
                                client.change_visual_shape(
                                    ray.object_unique_id,
                                    ray.link_index,
                                    &ChangeVisualShapeOptions {
                                        rgba_color: Some(colors[current_color]),
                                        ..Default::default()
                                    },
                                )?;
                                current_color = (current_color + 1) % colors.len()
                            }
                        }
                        eprintln!("aaa")
                    }
                }
            }
        }
        if !client.is_connected() {
            return Ok(());
        }
    }
}

fn get_ray_from_to(cam: DebugVisualizerCamera, mouse_x: f32, mouse_y: f32) -> ([f64; 3], [f64; 3]) {
    let cam_pos = [
        cam.target[0] - cam.distance * cam.camera_forward[0],
        cam.target[1] - cam.distance * cam.camera_forward[1],
        cam.target[2] - cam.distance * cam.camera_forward[2],
    ];
    let far_plane = 10000.;
    let ray_forward = [
        cam.target[0] - cam_pos[0],
        cam.target[1] - cam_pos[1],
        cam.target[2] - cam_pos[2],
    ];
    let inv_len = far_plane * 1.0
        / ((ray_forward[0] * ray_forward[0]
            + ray_forward[1] * ray_forward[1]
            + ray_forward[2] * ray_forward[2])
            .sqrt());
    let ray_forward = [
        inv_len * ray_forward[0],
        inv_len * ray_forward[1],
        inv_len * ray_forward[2],
    ];
    let ray_from = cam_pos;
    let one_over_width = 1.0 / cam.width as f32;
    let one_over_height = 1.0 / cam.height as f32;
    let d_hor = [
        cam.horizontal[0] * one_over_width,
        cam.horizontal[1] * one_over_width,
        cam.horizontal[2] * one_over_width,
    ];
    let d_ver = [
        cam.vertical[0] * one_over_height,
        cam.vertical[1] * one_over_height,
        cam.vertical[2] * one_over_height,
    ];
    let ray_to_center = [
        ray_from[0] + ray_forward[0],
        ray_from[1] + ray_forward[1],
        ray_from[2] + ray_forward[2],
    ];
    let ray_to = [
        ray_to_center[0] - 0.5 * cam.horizontal[0] + 0.5 * cam.vertical[0] + mouse_x * d_hor[0]
            - mouse_y * d_ver[0],
        ray_to_center[1] - 0.5 * cam.horizontal[1] + 0.5 * cam.vertical[1] + mouse_x * d_hor[1]
            - mouse_y * d_ver[1],
        ray_to_center[2] - 0.5 * cam.horizontal[2] + 0.5 * cam.vertical[2] + mouse_x * d_hor[2]
            - mouse_y * d_ver[2],
    ]
    .map(|i| i as f64);
    (ray_from.map(|i| i as f64), ray_to)
}
