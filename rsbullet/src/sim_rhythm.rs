use std::future::Future;
use std::time::{Duration, Instant};

use robot_behavior::PhysicsEngine;
use roplat::RoplatError;
use roplat::rhythm::Rhythm;

use crate::RsBullet;

/// 仿真步进节律源
///
/// 按固定周期驱动 `RsBullet` 物理引擎步进，每个 tick 调用 `engine.step()`
/// 后 yield `()` 给下游节点（可用于挂载观测/记录等节点）。
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

impl Rhythm for SimRhythm {
    type Yield = ();
    type Feed = ();
    type Output = ();
    type Error = RoplatError;

    async fn drive<N, F, Fut>(&mut self, mut nodes: N, mut op_domain: F)
    where
        N: Send,
        F: FnMut(N, Self::Yield) -> Fut + Send,
        Fut: Future<Output = (Self::Feed, N)> + Send,
    {
        let mut sequence = 0u32;
        let start_time = Instant::now();

        loop {
            let next_target = start_time + self.interval * sequence;
            tokio::time::sleep_until(next_target.into()).await;
            sequence += 1;

            let _ = self.engine.step();

            let ((), returned_nodes) = op_domain(nodes, ()).await;
            nodes = returned_nodes;
        }
    }
}
