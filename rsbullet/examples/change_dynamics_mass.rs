use std::{thread::sleep, time::Duration};

use nalgebra as na;
use rsbullet::{BulletResult, DynamicsUpdate, Mode, PhysicsClient, UrdfOptions};

fn main() -> BulletResult<()> {
    let mut client = PhysicsClient::connect(Mode::Gui)?;
    client
        .set_default_search_path()?
        .set_gravity([0., 0., -10.])?
        .set_time_step(Duration::from_secs_f64(1. / 240.))?;

    client.load_urdf(
        "cube.urdf",
        Some(UrdfOptions {
            base: Some(na::Isometry3::translation(0., 0., 3.)),
            use_fixed_base: true,
            ..Default::default()
        }),
    )?;
    let cube_2 = client.load_urdf(
        "cube.urdf",
        Some(UrdfOptions { use_fixed_base: true, ..Default::default() }),
    )?;

    client.change_dynamics(
        cube_2,
        -1,
        &DynamicsUpdate { mass: Some(1.), ..Default::default() },
    )?;

    while client.is_connected() {
        client.step_simulation()?;
        sleep(Duration::from_secs_f64(1. / 240.));
    }
    Ok(())
}
