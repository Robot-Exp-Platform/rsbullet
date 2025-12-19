use std::{sync::mpsc, time::Duration};

use robot_behavior::{
    AddCollision, AddRobot, AddSearchPath, AddVisual, PhysicsEngine, Renderer, RobotFile,
};
use rsbullet_core::{BulletResult, Mode, PhysicsClient};

use crate::{
    rsbullet_robot::{RsBulletRobot, RsBulletRobotBuilder},
    types::QueuedControl,
};

pub struct RsBullet {
    pub client: PhysicsClient,
    command_tx: mpsc::Sender<QueuedControl>,
    command_rx: mpsc::Receiver<QueuedControl>,
    active_controls: Vec<QueuedControl>,
    time_step: Duration,
}

impl RsBullet {
    pub fn new(mode: Mode) -> BulletResult<Self> {
        let (command_tx, command_rx) = mpsc::channel();
        let mut rsbullet = RsBullet {
            client: PhysicsClient::connect(mode)?,
            command_tx,
            command_rx,
            active_controls: Vec::new(),
            time_step: Duration::from_secs_f64(1.0 / 240.0),
        };
        rsbullet.client.set_default_search_path()?;
        Ok(rsbullet)
    }

    pub(crate) fn client_mut(&mut self) -> &mut PhysicsClient {
        &mut self.client
    }

    pub(crate) fn command_sender(&self) -> mpsc::Sender<QueuedControl> {
        self.command_tx.clone()
    }

    fn drain_queued_commands(&mut self) {
        for control in self.command_rx.try_iter() {
            self.active_controls.push(control);
        }
    }

    fn process_active_controls(&mut self) -> BulletResult<()> {
        let mut index = 0;
        while index < self.active_controls.len() {
            if self.active_controls[index](&mut self.client, self.time_step)? {
                let _ = self.active_controls.swap_remove(index);
            } else {
                index += 1;
            }
        }
        Ok(())
    }
}

impl PhysicsEngine for RsBullet {
    type Error = rsbullet_core::BulletError;

    fn reset(&mut self) -> BulletResult<()> {
        while self.command_rx.try_recv().is_ok() {}
        self.active_controls.clear();
        self.client.reset_simulation()?;
        Ok(())
    }
    fn step(&mut self) -> BulletResult<()> {
        self.drain_queued_commands();
        self.process_active_controls()?;
        self.client.step_simulation()?;
        Ok(())
    }
    fn shutdown(self) {
        self.client.disconnect();
    }

    fn set_step_time(&mut self, dt: Duration) -> BulletResult<&mut Self> {
        self.client.set_time_step(dt)?;
        self.time_step = dt;
        Ok(self)
    }
    fn set_gravity(&mut self, gravity: impl Into<[f64; 3]>) -> BulletResult<&mut Self> {
        self.client.set_gravity(gravity)?;
        Ok(self)
    }
}

impl AddSearchPath for RsBullet {
    type Error = rsbullet_core::BulletError;

    fn add_search_path(
        &mut self,
        path: impl AsRef<std::path::Path>,
    ) -> Result<&mut Self, Self::Error> {
        self.client.set_additional_search_path(path)?;
        Ok(self)
    }
}

impl Renderer for RsBullet {}

impl AddRobot for RsBullet {
    type PR<R> = RsBulletRobot<R>;
    type RB<'a, R: RobotFile> = RsBulletRobotBuilder<'a, R>;
    fn robot_builder<R: RobotFile>(&mut self, _name: impl ToString) -> RsBulletRobotBuilder<'_, R> {
        RsBulletRobotBuilder::new(self)
    }
}

impl AddCollision for RsBullet {
    type EntityId = crate::CollisionId;
    type CB<'a> = crate::EntityBuilder<'a, crate::CollisionMarker>;
    fn collision<'a>(&'a mut self, collision: robot_behavior::Collision<'a>) -> Self::CB<'a> {
        crate::EntityBuilder::collision(self, collision)
    }
}

impl AddVisual for RsBullet {
    type EntityId = crate::VisualId;
    type VB<'a> = crate::EntityBuilder<'a, crate::VisualMarker>;
    fn visual<'a>(&'a mut self, visual: robot_behavior::Visual<'a>) -> Self::VB<'a> {
        crate::EntityBuilder::visual(self, visual)
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use nalgebra as na;
    use robot_behavior::{AddCollision, AddRobot, AddVisual, Entity, EntityBuilder, PhysicsEngine};
    use rsbullet_core::{LoadModelFlags, Mode};

    struct ExRobot<const N: usize>;
    impl robot_behavior::RobotFile for ExRobot<6> {
        const URDF: &'static str = "robot.urdf";
    }

    use crate::{RsBullet, rsbullet_robot::RsBulletRobotBuilder};

    #[test]
    fn add_robot() {
        let mut engine = RsBullet::new(Mode::Direct).unwrap();
        let mut _robot1 = engine
            .robot_builder::<ExRobot<6>>("robot1")
            .base(na::Isometry3::identity())
            .base_fixed(true)
            .scaling(1.0)
            .load()
            .unwrap();

        let mut _robot1 = engine
            .robot_builder::<ExRobot<6>>("robot1")
            .base(na::Isometry3::identity())
            .base_fixed(true)
            .scaling(1.0)
            .load()
            .unwrap();

        let mut _robot2 = RsBulletRobotBuilder::<ExRobot<6>>::new(&mut engine)
            .base(na::Isometry3::identity())
            .base_fixed(true)
            .scaling(1.0)
            .load()
            .unwrap();

        let _robot3 = RsBulletRobotBuilder::<ExRobot<6>>::new(&mut engine)
            .flags(LoadModelFlags::URDF_ENABLE_CACHED_GRAPHICS_SHAPES)
            .use_maximal_coordinates(true)
            .load()
            .unwrap();

        let _ = (&mut _robot1, &mut _robot2, &_robot3, engine);
    }

    #[test]
    fn add_collision_and_visual() -> anyhow::Result<()> {
        let mut engine = RsBullet::new(Mode::Gui)?;

        engine
            .set_gravity([0., 0., -9.8])?
            .set_step_time(Duration::from_secs_f64(1. / 100.))?;

        let shift = [0., -0.02, 0.];
        let mesh_scale = [0.1, 0.1, 0.1];

        let _visual_shape = engine
            .visual(Entity::MeshFile { file: "duck.obj", scale: mesh_scale })
            .base(shift)
            .load()?;
        let _collision_shape = engine
            .collision(Entity::MeshFile { file: "duck_vhacd.obj", scale: mesh_scale })
            .base(shift)
            .load()?;

        loop {
            engine.step()?;
        }
    }
}
