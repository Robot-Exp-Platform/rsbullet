use std::{
    sync::{Arc, mpsc},
    time::Duration,
};

use roplat_behavior::{
    AddCollision, AddRobot, AddSearchPath, AddVisual, ModelFormat, PhysicsEngine, Renderer,
    RobotDescription, SimulationBackend, msgs::ManualClock,
};
use rsbullet_core::{BulletResult, Mode, PhysicsClient};

use crate::{
    rsbullet_robot::{RsBulletRobot, RsBulletRobotBuilder},
    types::{QueuedControl, StateRefresh, WorldValidity},
};

pub struct RsBullet {
    /// Low-level scene and query access for advanced applications.
    ///
    /// While using managed robot handles, call [`PhysicsEngine::reset`] on this
    /// engine to reset the world. Direct client resets, body removal, independent
    /// integration, or timestep changes bypass handle validity, cached state and
    /// counters, and are unsupported for a managed world.
    pub client: PhysicsClient,
    command_tx: mpsc::Sender<QueuedControl>,
    command_rx: mpsc::Receiver<QueuedControl>,
    active_controls: Vec<QueuedControl>,
    state_refresh: Vec<StateRefresh>,
    time_step: Duration,
    steps_completed: u64,
    simulated_time: Duration,
    clock: Arc<ManualClock>,
    faulted: bool,
    world_validity: WorldValidity,
}

impl RsBullet {
    pub fn new(mode: Mode) -> BulletResult<Self> {
        let (command_tx, command_rx) = mpsc::channel();
        let mut rsbullet = RsBullet {
            client: PhysicsClient::connect(mode)?,
            command_tx,
            command_rx,
            active_controls: Vec::new(),
            state_refresh: Vec::new(),
            time_step: Duration::from_secs_f64(1.0 / 240.0),
            steps_completed: 0,
            simulated_time: Duration::ZERO,
            clock: Arc::new(ManualClock::simulation()),
            faulted: false,
            world_validity: WorldValidity::new(),
        };
        rsbullet.client.set_default_search_path()?;
        rsbullet.client.set_time_step(rsbullet.time_step)?;
        Ok(rsbullet)
    }

    /// Internal native access. World lifecycle changes must go through the engine.
    pub(crate) fn client_mut(&mut self) -> &mut PhysicsClient {
        &mut self.client
    }

    pub(crate) fn command_sender(&self) -> mpsc::Sender<QueuedControl> {
        self.command_tx.clone()
    }

    pub(crate) fn world_validity(&self) -> std::sync::Arc<std::sync::atomic::AtomicBool> {
        self.world_validity.0.clone()
    }

    pub(crate) fn register_state_refresh(&mut self, refresh: StateRefresh) {
        self.state_refresh.push(refresh);
    }

    /// Observe the native world after every completed physics integration.
    ///
    /// Observers run in registration order, including internal robot-cache refreshes,
    /// before `step` returns and before `SimRhythm` emits its downstream tick. Use
    /// this for object poses, contacts, and evidence snapshots. `PhysicsClient`
    /// requires mutable access for queries; callbacks must not integrate or reset
    /// the world themselves. An error latches the engine fault and aborts the step.
    /// The completed native integration is still counted. Reset removes observers.
    pub fn add_step_observer<F>(&mut self, observer: F) -> &mut Self
    where
        F: FnMut(&mut PhysicsClient) -> BulletResult<()> + Send + 'static,
    {
        self.register_state_refresh(Box::new(observer));
        self
    }

    /// Native integrations completed since creation or the most recent reset.
    /// A failed post-step state refresh still counts the integration that occurred.
    pub fn steps_completed(&self) -> u64 {
        self.steps_completed
    }

    /// Sum of the configured timesteps of completed native integrations.
    pub fn simulated_time(&self) -> Duration {
        self.simulated_time
    }

    pub fn step_time(&self) -> Duration {
        self.time_step
    }

    /// Simulation clock of this world, reading [`RsBullet::simulated_time`].
    ///
    /// It moves after each completed integration, once the post-step observers
    /// have run, so a node waking on it reads the refreshed state. Stamp data
    /// with it and wait on it to run a graph in simulated time. Reset starts a
    /// new clock at zero; a clock taken before keeps the old world's time and
    /// no longer moves.
    pub fn clock(&self) -> Arc<ManualClock> {
        self.clock.clone()
    }

    /// A failed step requires resetting or replacing the world before continuing.
    /// Commands applied before an error cannot be rolled back.
    pub fn is_faulted(&self) -> bool {
        self.faulted
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
                // Retain submission order for the remaining controllers.
                let _ = self.active_controls.remove(index);
            } else {
                index += 1;
            }
        }
        Ok(())
    }

    fn step_inner(&mut self) -> BulletResult<()> {
        let next_step = self.steps_completed.checked_add(1).ok_or(
            rsbullet_core::BulletError::CommandFailed {
                message: "simulation step counter overflow",
                code: -1,
            },
        )?;
        let next_time = self.simulated_time.checked_add(self.time_step).ok_or(
            rsbullet_core::BulletError::CommandFailed {
                message: "simulation time overflow",
                code: -1,
            },
        )?;
        self.drain_queued_commands();
        self.process_active_controls()?;
        self.client.step_simulation()?;
        self.steps_completed = next_step;
        self.simulated_time = next_time;
        let refreshed = self
            .state_refresh
            .iter_mut()
            .try_for_each(|refresh| refresh(&mut self.client));
        // The integration happened even if a refresh failed.
        self.clock
            .set(i64::try_from(next_time.as_nanos()).unwrap_or(i64::MAX));
        refreshed
    }
}

impl SimulationBackend for RsBullet {
    const BACKEND_ID: &'static str = "bullet";
    const MODEL_FORMATS: &'static [ModelFormat] = &[ModelFormat::Urdf];
}

impl PhysicsEngine for RsBullet {
    type Error = rsbullet_core::BulletError;

    fn reset(&mut self) -> BulletResult<()> {
        // Existing robot handles refer to the old world and must be replaced.
        self.faulted = true;
        self.world_validity.invalidate();
        let (command_tx, command_rx) = mpsc::channel();
        self.command_tx = command_tx;
        self.command_rx = command_rx;
        self.active_controls.clear();
        self.state_refresh.clear();
        self.client.reset_simulation()?;
        self.client.set_time_step(self.time_step)?;
        self.steps_completed = 0;
        self.simulated_time = Duration::ZERO;
        self.clock = Arc::new(ManualClock::simulation());
        self.faulted = false;
        self.world_validity = WorldValidity::new();
        Ok(())
    }

    /// Apply queued controls, integrate once, then refresh every loaded robot's state.
    /// A failed command/integration/refresh latches a fault until reset. No later
    /// integration is silently attempted after a partial failure.
    fn step(&mut self) -> BulletResult<()> {
        if self.faulted {
            return Err(rsbullet_core::BulletError::CommandFailed {
                message: "simulation is faulted; reset and reload the world before stepping",
                code: -1,
            });
        }
        let outcome = self.step_inner();
        if outcome.is_err() {
            self.faulted = true;
            self.world_validity.invalidate();
        }
        outcome
    }
    fn shutdown(self) {
        self.client.disconnect();
    }

    fn set_step_time(&mut self, dt: Duration) -> BulletResult<&mut Self> {
        if dt.is_zero() {
            return Err(rsbullet_core::BulletError::CommandFailed {
                message: "simulation timestep must be positive",
                code: -1,
            });
        }
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
    type RB<'a, R: RobotDescription> = RsBulletRobotBuilder<'a, R>;
    fn robot_builder<R: RobotDescription>(
        &mut self,
        _name: impl ToString,
    ) -> RsBulletRobotBuilder<'_, R> {
        RsBulletRobotBuilder::new(self)
    }
}

impl AddCollision for RsBullet {
    type EntityId = crate::CollisionId;
    type CB<'a> = crate::EntityBuilder<'a, crate::CollisionMarker>;
    fn collision<'a>(&'a mut self, collision: roplat_behavior::Collision<'a>) -> Self::CB<'a> {
        crate::EntityBuilder::collision(self, collision)
    }
}

impl AddVisual for RsBullet {
    type EntityId = crate::VisualId;
    type VB<'a> = crate::EntityBuilder<'a, crate::VisualMarker>;
    fn visual<'a>(&'a mut self, visual: roplat_behavior::Visual<'a>) -> Self::VB<'a> {
        crate::EntityBuilder::visual(self, visual)
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use nalgebra as na;
    use roplat_behavior::{
        AddCollision, AddRobot, AddVisual, Entity, EntityBuilder, PhysicsEngine,
    };
    use rsbullet_core::{LoadModelFlags, Mode};

    struct ExRobot<const N: usize>;
    impl roplat_behavior::RobotDescription for ExRobot<6> {
        const URDF: Option<&'static str> = Some(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/assets/single_axis.urdf"
        ));
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

    // Interactive demonstration: opens a GUI and steps forever; exclude from CI.
    #[test]
    #[ignore = "interactive GUI demonstration with an unbounded simulation loop"]
    fn add_collision_and_visual() -> anyhow::Result<()> {
        let mut engine = RsBullet::new(Mode::Gui)?;

        engine
            .set_gravity([0., 0., -9.8])?
            .set_step_time(Duration::from_secs_f64(1. / 100.))?;

        let shift = [0., -0.02, 0.];
        let mesh_scale = [0.1, 0.1, 0.1];

        let _visual_shape = engine
            .visual(Entity::MeshFile {
                file: "duck.obj",
                scale: mesh_scale,
            })
            .base(shift)
            .load()?;
        let _collision_shape = engine
            .collision(Entity::MeshFile {
                file: "duck_vhacd.obj",
                scale: mesh_scale,
            })
            .base(shift)
            .load()?;

        loop {
            engine.step()?;
        }
    }
}
