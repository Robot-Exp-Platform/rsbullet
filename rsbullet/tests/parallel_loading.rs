//! Worlds on different threads may load meshed models at the same time.
//!
//! Bullet keeps the meshes it reads from OBJ files in one process-wide table
//! without a lock; before rsbullet-core serialised the commands that read
//! meshes, loading the Panda (OBJ collision and visual meshes) in two worlds
//! at once crashed the process with an access violation or hung it.

use rsbullet::{Mode, RsBullet, UrdfOptions};

const PANDA_DATA: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../rsbullet-sys/bullet3/examples/pybullet/gym/pybullet_data"
);

#[test]
fn eight_threads_load_the_panda_at_once() {
    for _round in 0..3 {
        let threads: Vec<_> = (0..8)
            .map(|_| {
                std::thread::spawn(|| {
                    let mut world = RsBullet::new(Mode::Direct).unwrap();
                    world.client.set_additional_search_path(PANDA_DATA).unwrap();
                    let options = UrdfOptions { use_fixed_base: true, ..Default::default() };
                    let panda = world
                        .client
                        .load_urdf("franka_panda/panda.urdf", Some(options))
                        .unwrap();
                    world.client.get_num_joints(panda)
                })
            })
            .collect();
        for thread in threads {
            assert_eq!(thread.join().expect("a loading thread panicked"), 12);
        }
    }
}
