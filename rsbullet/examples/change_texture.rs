use std::time::Instant;

use rsbullet::{
    BulletResult, CameraImageOptions, ChangeVisualShapeOptions, LoggingType, Mode, PhysicsClient,
    Renderer, TextureData,
};

fn main() -> BulletResult<()> {
    let mut client = PhysicsClient::connect(Mode::Gui)?;
    client
        .set_default_search_path()?
        .set_gravity([0., 0., -10.])?;

    let plane_a = client.load_urdf("plane.urdf", None::<()>)?;
    let plane = client.load_urdf("cube.urdf", Some([0., 0., 1.]))?;
    let texture = client.load_texture("tex256.png")?;

    client
        .change_visual_shape(
            plane_a,
            -1,
            &ChangeVisualShapeOptions {
                rgba_color: Some([1., 13., 1., 0.5]),
                ..Default::default()
            },
        )?
        .change_visual_shape(
            plane,
            -1,
            &ChangeVisualShapeOptions { rgba_color: Some([1., 1., 1., 0.5]), ..Default::default() },
        )?
        .change_visual_shape(
            plane,
            -1,
            &ChangeVisualShapeOptions { rgba_color: Some([1., 1., 1., 0.5]), ..Default::default() },
        )?;

    const WIDTH: usize = 256;
    const HEIGHT: usize = 256;
    let mut pixels = [255u8; WIDTH * HEIGHT * 3];
    let mut blue = 0;

    let log_id =
        client.start_state_logging(LoggingType::ProfileTimings, "renderbench.json", None::<()>)?;

    for _ in 0..100_000 {
        client.step_simulation()?;
        for i in 0..WIDTH {
            for j in 0..HEIGHT {
                let idx = (i + j * WIDTH) * 3;
                pixels[idx] = (i % 256) as u8;
                pixels[idx + 1] = ((j + blue) % 256) as u8;
                pixels[idx + 2] = blue as u8;
            }
        }
        blue = (blue + 1) % 256;
        client.change_texture(
            texture,
            &TextureData {
                width: WIDTH as i32,
                height: HEIGHT as i32,
                rgb_pixels: &pixels,
            },
        )?;

        let start_time = Instant::now();
        client.get_camera_image(&CameraImageOptions {
            width: 300,
            height: 300,
            renderer: Some(Renderer::BulletHardwareOpenGl),
            ..Default::default()
        })?;
        print!("rendering time: {:?}\r", start_time.elapsed());
    }
    client.stop_state_logging(log_id)?;
    Ok(())
}
