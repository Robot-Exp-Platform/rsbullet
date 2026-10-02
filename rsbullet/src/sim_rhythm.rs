use std::future::Future;
use std::time::{Duration, Instant};

use robot_behavior::PhysicsEngine;
use roplat::rhythm::Rhythm;
use roplat::{Completion, Execution, ExecutionContext, Lifecycle, RoplatError};

use crate::RsBullet;

/// Rhythm for stepping a simulation.
///
/// Drives the `RsBullet` physics engine at a fixed period and calls
/// `engine.step()` on every tick. It yields `()` to downstream nodes.
///
/// ```ignore
/// let mut sim_rhythm = SimRhythm::new(engine, Duration::from_secs_f64(1.0 / 240.0));
/// sim_rhythm >> { /* observer nodes */ };
/// ```
pub struct SimRhythm {
    engine: RsBullet,
    interval: Duration,
}

impl SimRhythm {
    pub fn new(engine: RsBullet, interval: Duration) -> Self {
        Self { engine, interval }
    }

    /// 获取引擎的可变引用（用于初始化后的额外配置）
    pub fn engine_mut(&mut self) -> &mut RsBullet {
        &mut self.engine
    }
}

impl Lifecycle for SimRhythm {
    type Error = RoplatError;
}

impl Rhythm for SimRhythm {
    type Yield = ();
    type Feed = ();
    type Input = ();
    type Output = ();

    async fn drive<N, F, Fut>(
        &mut self,
        mut nodes: N,
        mut op_domain: F,
        _input: Self::Input,
        context: ExecutionContext,
    ) -> (Execution<()>, N)
    where
        N: Send,
        F: FnMut(N, Self::Yield, ExecutionContext) -> Fut + Send,
        Fut: Future<Output = (Execution<Self::Feed>, N)> + Send,
    {
        let mut sequence = 0u32;
        let start_time = Instant::now();

        loop {
            if context.is_stopping() {
                return (Ok(Completion::Stopped), nodes);
            }
            let next_target = start_time + self.interval * sequence;
            tokio::time::sleep_until(next_target.into()).await;
            sequence += 1;

            if context.is_stopping() {
                return (Ok(Completion::Stopped), nodes);
            }
            // Existing simulator policy is intentionally unchanged: step errors
            // are ignored until simulator semantics receive their own review.
            let _ = self.engine.step();

            let (outcome, returned_nodes) = op_domain(nodes, (), context.clone()).await;
            nodes = returned_nodes;
            match outcome {
                Ok(Completion::Completed(())) => {}
                Ok(Completion::Stopped) => return (Ok(Completion::Stopped), nodes),
                Err(error) => {
                    context.request_stop();
                    return (Err(error), nodes);
                }
            }
        }
    }
}
