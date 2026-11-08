mod support;

use std::{
    ops::ControlFlow,
    time::{Duration, Instant},
};

use roplat_behavior::behavior::{Arm, JointSpace};
use roplat_behavior::msgs::Clock;
use roplat_behavior::{
    AsyncControl, Control, ControlExit, JointPositionControl, JointState, JointVelocityControl,
    MoveTo, Robot, RobotException,
};
use roplat_world::PhysicsEngine;
use rsbullet::{
    ARRIVAL_TIMEOUT, ARRIVAL_WINDOW, DEFAULT_CONTROL_PERIOD, Mode, RsBullet, RsBulletDevice,
    UrdfOptions,
};
use support::{Axis, device, load, position};

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
    device.control_with_flow::<JointPositionControl<1>, _>(|_, _| {
        ControlFlow::Break(ControlExit::Stopped)
    })?;
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
fn move_to_plans_with_the_motion_limits() -> anyhow::Result<()> {
    let mut full = device()?;
    MoveTo::<JointSpace<1>>::move_to(&mut full, [0.4])?;

    let mut slow = device()?;
    let half = Arm::<1>::motion_limits(&slow).scaled(0.5);
    Arm::<1>::set_motion_limits(&mut slow, half)?;
    MoveTo::<JointSpace<1>>::move_to(&mut slow, [0.4])?;
    let (full_steps, slow_steps) = (
        full.engine().steps_completed(),
        slow.engine().steps_completed(),
    );
    assert!(slow_steps > full_steps, "{slow_steps} <= {full_steps}");
    let q = position(&mut slow)?;
    assert!((q - 0.4).abs() < 0.02, "q = {q}");

    // Limits above the declared envelope are refused.
    let mut loose = half;
    loose.joint.dq_max = [2.0];
    assert!(Arm::<1>::set_motion_limits(&mut slow, loose).is_err());
    assert_eq!(Arm::<1>::motion_limits(&slow), half);
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

#[test]
fn the_device_reports_its_model_control_period() {
    struct Kilohertz;
    impl rsbullet::DeviceModel for Kilohertz {
        const CONTROL_PERIOD: Duration = Duration::from_millis(1);
    }
    assert_eq!(
        <RsBulletDevice<Kilohertz> as Robot>::CONTROL_PERIOD,
        Duration::from_millis(1)
    );
    // A model that names no period keeps the earlier constant, 1/240 s.
    assert_eq!(
        <RsBulletDevice<Axis> as Robot>::CONTROL_PERIOD,
        DEFAULT_CONTROL_PERIOD
    );
    assert_eq!(DEFAULT_CONTROL_PERIOD, Duration::from_nanos(4_166_666));
    assert_eq!(
        <rsbullet::RsBulletRobot<Axis> as Robot>::CONTROL_PERIOD,
        DEFAULT_CONTROL_PERIOD
    );
}

#[test]
fn a_position_session_drives_no_joint_faster_than_its_speed_limit() -> anyhow::Result<()> {
    // The axis may move at 1 m/s; its 100 N motor on 1 kg could go far
    // faster towards a target 0.9 m away.
    let mut device = device()?;
    let mut fastest = 0.0_f64;
    let mut cycles = 0;
    device.control_with::<JointPositionControl<1>, _>(|state, _| {
        let dq = state.meas.dq().expect("the speed is measured");
        fastest = fastest.max(dq[0].abs());
        cycles += 1;
        ([0.9], cycles == 480)
    })?;
    assert!(
        fastest > 0.95 && fastest <= 1.001,
        "peak speed {fastest} m/s"
    );
    let q = position(&mut device)?;
    assert!((q - 0.9).abs() < 0.01, "q = {q}");
    Ok(())
}

#[test]
fn move_to_returns_with_every_joint_in_position() -> anyhow::Result<()> {
    let mut device = device()?;
    MoveTo::<JointSpace<1>>::move_to(&mut device, [-0.7])?;
    let q = position(&mut device)?;
    assert!((q + 0.7).abs() <= ARRIVAL_WINDOW, "q = {q}");
    for _ in 0..240 {
        device.engine_mut().step()?;
    }
    let held = position(&mut device)?;
    assert!((held + 0.7).abs() <= ARRIVAL_WINDOW, "held at {held}");
    Ok(())
}

#[test]
fn a_blocked_move_fails_after_the_arrival_timeout_and_stops_there() -> anyhow::Result<()> {
    let mut engine = RsBullet::new(Mode::Direct)?;
    let robot = load(&mut engine, 0.0)?;
    // A fixed 0.2 m block centred at x = 0.5 stops the 0.1 m carriage at
    // about x = 0.35.
    let block = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/assets/camera_block.urdf"
    );
    let options = UrdfOptions {
        base: Some(nalgebra::Isometry3::translation(0.5, 0.0, 0.0)),
        use_fixed_base: true,
        ..Default::default()
    };
    engine.client.load_urdf(block, Some(options))?;
    let mut device = RsBulletDevice::new(engine, robot)?;
    let t0 = device.engine().simulated_time();
    let outcome = MoveTo::<JointSpace<1>>::move_to(&mut device, [0.9]);
    let elapsed = device.engine().simulated_time() - t0;
    assert!(
        matches!(outcome, Err(RobotException::CommandException(_))),
        "{outcome:?}"
    );
    assert!(elapsed > ARRIVAL_TIMEOUT, "failed after {elapsed:?}");
    let stopped = position(&mut device)?;
    assert!((stopped - 0.35).abs() < 0.02, "stopped at {stopped}");
    for _ in 0..240 {
        device.engine_mut().step()?;
    }
    let held = position(&mut device)?;
    assert!((held - stopped).abs() < 1e-3, "{stopped} -> {held}");
    Ok(())
}
