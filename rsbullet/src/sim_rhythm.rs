//! Bullet-compatible names for the shared simulator execution adapter.
//!
//! The implementation lives in roplat_world so another simulator can reuse
//! the same scheduler without linking Bullet. The default engine type preserves
//! the existing `rsbullet::SimRhythm` API.

pub use roplat_world::SimPacing;

/// Shared finite/paced simulator rhythm, defaulting to this crate's native engine.
/// See [`roplat_world::SimulationRhythm`] for execution semantics.
pub type SimRhythm<E = crate::RsBullet> = roplat_world::SimulationRhythm<E>;
