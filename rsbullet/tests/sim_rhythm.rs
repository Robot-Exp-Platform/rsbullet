#![cfg(feature = "roplat")]

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use roplat::RoplatResult;
use roplat_behavior::roplat::SimulationRhythm;
use rsbullet::{Mode, RsBullet, SimRhythm};

#[roplat::system]
async fn run_native(mut rhythm: SimRhythm) -> RoplatResult<()> {
    rhythm >> {};
    Ok(rhythm.output)
}

// Generic scheduler semantics live in roplat_behavior/tests/simulation_rhythm.rs.
// This test proves the compatibility alias still drives the actual native engine
// through the System macro, with post-integration observations for each tick.
#[tokio::test]
async fn shared_rhythm_alias_drives_native_bullet_in_a_system() -> anyhow::Result<()> {
    let mut engine = RsBullet::new(Mode::Direct)?;
    let calls = Arc::new(AtomicUsize::new(0));
    let captured = calls.clone();
    engine.add_step_observer(move |_| {
        captured.fetch_add(1, Ordering::SeqCst);
        Ok(())
    });
    let rhythm: SimulationRhythm<RsBullet> = SimRhythm::logical(engine).with_max_steps(3);
    run_native(rhythm).await?;
    assert_eq!(calls.load(Ordering::SeqCst), 3);
    Ok(())
}
