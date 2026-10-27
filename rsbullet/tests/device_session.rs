mod support;

use std::{
    ops::ControlFlow,
    time::{Duration, Instant},
};

use roplat_behavior::behavior::JointSpace;
use roplat_behavior::msgs::Clock;
use roplat_behavior::{
    AsyncControl, Control, JointPositionControl, JointState, JointVelocityControl, MoveTo,
    PhysicsEngine,
};
use rsbullet::{Mode, RsBullet, RsBulletDevice};
use support::{device, load, position};

#[test]
fn every_cycle_integrates_once_with_the_physics_timestep() -> anyhow::Result<()> {
    let mut device = device()?;
    let dt = device.engine().step_time();
    let mut periods = Vec::new();
    device.control_with::<JointPositionControl<1>, _>(|_, period| {
        periods.push(period);
        ([0.5], periods.len() == 720)
    })?;
    assert_eq!(device.engine().steps_completed(), 720);
    assert!(periods.iter().all(|&period| period == dt));
    let q = position(&mut device)?;
    assert!((q - 0.5).abs() < 0.02, "q = {q}");
    assert_eq!(
        device.engine().clock().now().as_duration(),
        Some(device.engine().simulated_time())
    );
    Ok(())
}

#[test]
fn break_applies_nothing_and_done_integrates_its_command() -> anyhow::Result<()> {
    let mut device = device()?;
    device.control_with_flow::<JointPositionControl<1>, _>(|_, _| ControlFlow::Break(()))?;
    assert_eq!(device.engine().steps_completed(), 0);
    device.control_with::<JointPositionControl<1>, _>(|_, _| ([0.5], true))?;
    assert_eq!(device.engine().steps_completed(), 1);
    Ok(())
}

#[test]
fn a_finished_velocity_session_leaves_the_joint_holding() -> anyhow::Result<()> {
    let mut device = device()?;
    let mut cycles = 0;
    device.control_with::<JointVelocityControl<1>, _>(|_, _| {
        cycles += 1;
        ([0.5], cycles == 120)
    })?;
    let released = position(&mut device)?;
    assert!(released > 0.1, "the session moved the joint: {released}");
    for _ in 0..240 {
        device.engine_mut().step()?;
    }
    let held = position(&mut device)?;
    assert!((held - released).abs() < 0.01, "{released} -> {held}");
    Ok(())
}

#[test]
fn a_device_rejects_a_robot_from_another_world() -> anyhow::Result<()> {
    let mut first = RsBullet::new(Mode::Direct)?;
    let robot = load(&mut first, 0.0)?;
    let second = RsBullet::new(Mode::Direct)?;
    assert!(RsBulletDevice::new(second, robot).is_err());
    Ok(())
}

#[test]
fn queued_controls_of_other_bodies_run_during_a_session() -> anyhow::Result<()> {
    let mut engine = RsBullet::new(Mode::Direct)?;
    let arm = load(&mut engine, 0.0)?;
    let mut neighbour = load(&mut engine, 1.0)?;
    MoveTo::<JointSpace<1>>::move_to(&mut neighbour, [0.3])?;
    let mut device = RsBulletDevice::new(engine, arm)?;
    let mut cycles = 0;
    device.control_with::<JointPositionControl<1>, _>(|_, _| {
        cycles += 1;
        ([0.0], cycles == 720)
    })?;
    let q = position(&mut neighbour)?;
    assert!((q - 0.3).abs() < 0.02, "neighbour q = {q}");
    Ok(())
}

#[test]
fn move_to_returns_after_the_motion_is_integrated() -> anyhow::Result<()> {
    let mut device = device()?;
    MoveTo::<JointSpace<1>>::move_to(&mut device, [0.4])?;
    assert!(device.engine().steps_completed() > 0);
    let q = position(&mut device)?;
    assert!((q - 0.4).abs() < 0.02, "q = {q}");
    Ok(())
}

#[test]
fn the_clock_follows_integrations_and_restarts_on_reset() -> anyhow::Result<()> {
    let mut engine = RsBullet::new(Mode::Direct)?;
    let clock = engine.clock();
    for _ in 0..3 {
        engine.step()?;
    }
    let three = engine.step_time() * 3;
    assert_eq!(clock.now().as_duration(), Some(three));
    engine.reset()?;
    assert_eq!(engine.clock().now().as_duration(), Some(Duration::ZERO));
    assert_eq!(clock.now().as_duration(), Some(three));
    Ok(())
}

#[test]
fn realtime_sessions_wait_for_each_cycle() -> anyhow::Result<()> {
    let mut device = device()?.realtime(true);
    let dt = device.engine().step_time();
    let mut cycles = 0;
    let start = Instant::now();
    device.control_with::<JointPositionControl<1>, _>(|_, _| {
        cycles += 1;
        ([0.0], cycles == 24)
    })?;
    // The first cycle starts at once; each later one waits for its slot.
    assert!(start.elapsed() >= dt * 23);
    Ok(())
}

#[tokio::test]
async fn an_async_session_integrates_once_per_cycle() -> anyhow::Result<()> {
    let mut device = device()?;
    let mut cycles = 0;
    let mut controller = |_: JointState<1>, _: Duration| {
        cycles += 1;
        let done = cycles == 720;
        async move { ControlFlow::Continue(([0.5], done)) }
    };
    device
        .control_native_async::<JointPositionControl<1>, _>(&mut controller)
        .await?;
    assert_eq!(device.engine().steps_completed(), 720);
    let q = position(&mut device)?;
    assert!((q - 0.5).abs() < 0.02, "q = {q}");
    Ok(())
}

#[tokio::test]
async fn an_async_session_cannot_pace_on_the_wall_clock() -> anyhow::Result<()> {
    let mut device = device()?.realtime(true);
    let mut controller =
        |_: JointState<1>, _: Duration| async { ControlFlow::Continue(([0.0], true)) };
    let session = device
        .control_native_async::<JointPositionControl<1>, _>(&mut controller)
        .await;
    assert!(session.is_err());
    assert_eq!(device.engine().steps_completed(), 0);
    Ok(())
}
