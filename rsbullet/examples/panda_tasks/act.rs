//! Task steps as nodes, and how a failed step is reported.
//!
//! A step is a node that borrows the arm and the planner for one `process`
//! and returns a business `Result`: graphs decide what a failure means, by
//! raising it with `?`, retrying it, or recording it.
use std::fmt;

use roplat::{Completion, Execution, Lifecycle, Node, RoplatError};

use crate::world::{Device, Planner};

/// Why a step did not do what it should.
#[derive(Clone, Debug, PartialEq)]
pub enum Failure {
    /// No joint configuration reaches the pose.
    Unreachable(String),
    /// The pose is reachable but the hand would touch something there.
    Blocked(String),
    /// No collision-free path, or the path could not be timed.
    Plan(String),
    /// The follower refused: the arm left the path or missed the goal.
    Track(String),
    /// The fingers closed without the cube between them.
    Missed { width: f64 },
    /// The cube left the hand on the way.
    Slipped { width: f64 },
    /// The simulator or the device reported an error.
    Device(String),
    /// The step was stopped.
    Stopped,
}

impl fmt::Display for Failure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unreachable(why) => write!(f, "unreachable: {why}"),
            Self::Blocked(why) => write!(f, "blocked: {why}"),
            Self::Plan(why) => write!(f, "no plan: {why}"),
            Self::Track(why) => write!(f, "tracking: {why}"),
            Self::Missed { width } => write!(f, "missed the cube, fingers at {width:.4} m"),
            Self::Slipped { width } => write!(f, "the cube slipped, fingers at {width:.4} m"),
            Self::Device(why) => write!(f, "device: {why}"),
            Self::Stopped => write!(f, "stopped"),
        }
    }
}

impl std::error::Error for Failure {}

/// Raised with `?`, a failure travels as an I/O error so that an isolating
/// `.outcome` can recover it, as `ControlSessionError` does.
impl From<Failure> for RoplatError {
    fn from(failure: Failure) -> Self {
        RoplatError::Io(std::io::Error::other(failure))
    }
}

impl Failure {
    /// The value, or the failure that ended an isolated group of steps.
    pub fn from_outcome<T>(outcome: Execution<T>) -> Result<T, Failure> {
        match outcome {
            Ok(Completion::Completed(value)) => Ok(value),
            Ok(Completion::Stopped) => Err(Self::Stopped),
            Err(error) => Err(Self::of(&error)),
        }
    }

    fn of(error: &RoplatError) -> Self {
        if let RoplatError::Io(io) = error
            && let Some(failure) = io.get_ref().and_then(|e| e.downcast_ref::<Self>())
        {
            return failure.clone();
        }
        Self::Device(error.to_string())
    }

    /// Whether trying the same thing again can help.
    pub fn retryable(&self) -> bool {
        matches!(self, Self::Missed { .. })
    }
}

/// The work of one step: `step(arm, planner, input)`.
type StepFn<I, O> =
    Box<dyn FnMut(&mut Device, &mut Planner, I) -> Result<O, Failure> + Send + Sync>;

/// A task step: one call of its function per `process`.
pub struct Act<I, O> {
    name: &'static str,
    step: StepFn<I, O>,
}

pub fn act<I, O>(
    name: &'static str,
    step: impl FnMut(&mut Device, &mut Planner, I) -> Result<O, Failure> + Send + Sync + 'static,
) -> Act<I, O> {
    Act {
        name,
        step: Box::new(step),
    }
}

impl<I, O> Lifecycle for Act<I, O> {
    type Error = RoplatError;
}

impl<I: Send, O: Send> Node for Act<I, O> {
    type Resources = (Device, Planner);
    type Input = I;
    type Output = Result<O, Failure>;

    async fn process<'call>(
        &'call mut self,
        input: I,
        (arm, planner): (&'call mut Device, &'call mut Planner),
    ) -> Result<O, Failure> {
        let result = (self.step)(arm, planner, input);
        let at = arm.engine().simulated_time().as_secs_f64();
        match &result {
            Ok(_) => println!("{at:>8.2} s  {}", self.name),
            Err(failure) => println!("{at:>8.2} s  {}: {failure}", self.name),
        }
        result
    }
}
