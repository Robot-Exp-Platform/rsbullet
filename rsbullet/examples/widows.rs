use std::{thread::sleep, time::Duration};

use rsbullet::{BulletResult, CameraImageOptions, Mode, PhysicsClient, UrdfOptions};
fn main() -> BulletResult<()> {
    let mut client = PhysicsClient::connect(Mode::Gui)?;

    client
        .set_default_search_path()?
        .set_gravity([0., 0., -10.])?;

    client.load_urdf("table/table", Some([0.5, 0., -0.82]))?;
    let arm = client.load_urdf(
        "widowx/widowx.urdf",
        Some(UrdfOptions { use_fixed_base: true, ..Default::default() }),
    )?;

    client.reset_base_position_and_orientation(
        arm,
        [-0.098612, -0.000726, -0.194018],
        [0., 0., 0., 1.],
    )?;

    loop {
        client.step_simulation()?;
        sleep(Duration::from_secs_f64(0.01));
        let view_matrix = client.get_debug_visualizer_camera()?.view_matrix;
        let projection_matrix = client.get_debug_visualizer_camera()?.projection_matrix;
        client.get_camera_image(&CameraImageOptions {
            width: 640,
            height: 480,
            view_matrix: Some(view_matrix),
            projection_matrix: Some(projection_matrix),
            ..Default::default()
        })?;
    }
}
