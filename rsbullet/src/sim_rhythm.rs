//! Bullet-compatible names for the shared simulator execution adapter.
//!
//! The implementation lives in robot_behavior so another simulator can reuse
//! the same scheduler without linking Bullet. The default engine type preserves
//! the existing `rsbullet::SimRhythm` API.

pub use robot_behavior::roplat::SimPacing;

/// Shared finite/paced simulator rhythm, defaulting to this crate's native engine.
/// See [`robot_behavior::roplat::SimulationRhythm`] for execution semantics.
pub type SimRhythm<E = crate::RsBullet> = robot_behavior::roplat::SimulationRhythm<E>;
