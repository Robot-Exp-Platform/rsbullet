#![cfg(feature = "roplat")]

mod support;

use std::{ops::ControlFlow, time::Duration};

use roplat::RoplatResult;
use roplat_behavior::roplat::{AsyncControlRhythm, ControlRhythm};
use roplat_behavior::{
    AsyncControlCallback, AsyncControlWith, ControlStep, ControlWith, JointPositionControl,
    JointSample, JointState, Robot, RobotResult,
};
use support::{device, position};

/// One graph, written against the `ControlWith` contract only.
#[roplat::system]
async fn approach<R>(mut device: R, target: f64, cycles: usize) -> RoplatResult<R>
where
    R: ControlWith<JointPositionControl<1>> + Send,
{
    let mut control = ControlRhythm::<R, JointPositionControl<1>>::new();
    let mut done = 0;
    let mut servo = roplat::node::map(move |_: (JointState<1>, Duration)| {
        done += 1;
        ([target], done == cycles)
    });
    &mut device >> control.resources.0;
    control
        >> |sample| {
            sample >> servo;
            servo
        };
    Ok(device)
}

/// The same graph over the native asynchronous session.
#[roplat::system]
async fn approach_async<R>(mut device: R, target: f64, cycles: usize) -> RoplatResult<R>
where
    R: AsyncControlWith<JointPositionControl<1>> + Send,
{
    let mut control = AsyncControlRhythm::<R, JointPositionControl<1>>::new();
    let mut done = 0;
    let mut servo = roplat::node::map(move |_: (JointState<1>, Duration)| {
        done += 1;
        ([target], done == cycles)
    });
    &mut device >> control.resources.0;
    control
        >> |sample| {
            sample >> servo;
            servo
        };
    Ok(device)
}

/// Stands in for a real driver: it paces its own cycles and servos directly.
struct BenchArm {
    q: f64,
    cycles: usize,
}

impl Robot for BenchArm {
    type State = f64;
    const CONTROL_PERIOD: f64 = 0.001;

    fn version() -> String {
        "bench".into()
    }

    fn read_state(&mut self) -> RobotResult<f64> {
        Ok(self.q)
    }
}

impl ControlWith<JointPositionControl<1>> for BenchArm {
    fn hold_command(state: &JointState<1>) -> [f64; 1] {
        state.meas.q().unwrap_or([0.0])
    }

    fn control_with_flow<F>(&mut self, mut closure: F) -> RobotResult<()>
    where
        F: FnMut(JointState<1>, Duration) -> ControlStep<[f64; 1]>,
    {
        loop {
            let state = JointState::from_meas(JointSample::EMPTY.with_q([self.q]));
            let ControlFlow::Continue(([q], done)) = closure(state, Duration::from_millis(1))
            else {
                return Ok(());
            };
            self.q = q;
            self.cycles += 1;
            if done {
                return Ok(());
            }
        }
    }
}

impl AsyncControlWith<JointPositionControl<1>> for BenchArm {
    async fn control_native_async<C>(&mut self, callback: &mut C) -> RobotResult<()>
    where
        C: AsyncControlCallback<JointState<1>, [f64; 1]>,
    {
        loop {
            let state = JointState::from_meas(JointSample::EMPTY.with_q([self.q]));
            let ControlFlow::Continue(([q], done)) =
                callback.call(state, Duration::from_millis(1)).await
            else {
                return Ok(());
            };
            self.q = q;
            self.cycles += 1;
            if done {
                return Ok(());
            }
        }
    }
}

#[tokio::test]
async fn one_graph_drives_a_driver_and_the_simulation() -> anyhow::Result<()> {
    let bench = approach(BenchArm { q: 0.0, cycles: 0 }, 0.4, 5).await?;
    assert_eq!((bench.q, bench.cycles), (0.4, 5));

    let mut simulated = approach(device()?, 0.4, 720).await?;
    assert_eq!(simulated.engine().steps_completed(), 720);
    let q = position(&mut simulated)?;
    assert!((q - 0.4).abs() < 0.02, "q = {q}");
    Ok(())
}

#[tokio::test]
async fn one_async_graph_drives_a_driver_and_the_simulation() -> anyhow::Result<()> {
    let bench = approach_async(BenchArm { q: 0.0, cycles: 0 }, 0.4, 5).await?;
    assert_eq!((bench.q, bench.cycles), (0.4, 5));

    let mut simulated = approach_async(device()?, 0.4, 720).await?;
    assert_eq!(simulated.engine().steps_completed(), 720);
    let q = position(&mut simulated)?;
    assert!((q - 0.4).abs() < 0.02, "q = {q}");
    Ok(())
}
