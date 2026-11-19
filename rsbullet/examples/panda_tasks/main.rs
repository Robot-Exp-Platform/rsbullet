//! Task orchestration written as System graphs over a simulated Panda.
//!
//! cargo run --release -p rsbullet --no-default-features --features roplat \
//!   --example panda_tasks -- TASK /path/to/bullet3/examples/pybullet/gym/pybullet_data
//!
//! TASK is one of
//!
//! - `recover`: pick and place that falls back to another way of holding the
//!   cube when one is blocked, and retries a missed grasp;
//! - `batch`: the pick and place repeated from several spots, each failure
//!   recorded without stopping the batch;
//! - `sweep`: one motion repeated at several speeds;
//! - `contact`: a descent stopped by its control loop when the hand touches
//!   the table.
//!
//! Every step is a node that borrows the arm and the planning scene; the
//! order, the retries, the repetitions and the fallbacks are rhythms in
//! `#[roplat::system]` graphs. Each task fails unless its result is what the
//! task states. Build with `--release`: in a debug build COPP's solver takes
//! about half a minute per motion.
mod act;
mod batch;
mod contact;
mod recover;
mod steps;
mod sweep;
mod world;

use std::path::Path;

use world::{ABOVE, Cell, HOME, Planner, REST, Yaw};

const USAGE: &str = "expected TASK MODEL_DIRECTORY, TASK one of recover, batch, sweep, contact";

#[tokio::main(flavor = "current_thread")]
async fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let task = args.next().ok_or_else(|| anyhow::anyhow!(USAGE))?;
    let assets = args.next().ok_or_else(|| anyhow::anyhow!(USAGE))?;
    anyhow::ensure!(args.next().is_none(), USAGE);
    let assets = Path::new(&assets);
    match task.as_str() {
        "recover" => run_recover(assets).await,
        "batch" => run_batch(assets).await,
        "sweep" => run_sweep(assets).await,
        "contact" => run_contact(assets).await,
        other => anyhow::bail!("unknown task {other:?}; {USAGE}"),
    }
}

async fn run_contact(assets: &Path) -> anyhow::Result<()> {
    // The cube rests out of the way; the hand comes down on bare table.
    let cube_at = [0.30, 0.30];
    let spot = [0.55, -0.20];
    let cell = Cell::new(assets, cube_at, &[])?;
    let planner = Planner::new(assets, cube_at, &[])?;
    // A straight line from 10 cm above the table to 2 cm inside it.
    let mut path = Vec::new();
    let mut seed = HOME;
    for step in 0..=12 {
        let z = world::TABLE_TOP + 0.10 - 0.01 * f64::from(step);
        seed = planner.reach([spot[0], spot[1], z], Yaw::ALONG_Y, seed)?;
        path.push(seed);
    }
    let above = path[0];
    let descent =
        world::follower(&path, contact::DESCENT_SPEED, &cell.clock)?.with_path_tolerance([0.5; 7]);
    let (_, planner, touch, quiet) = contact::contact(
        cell.arm,
        planner,
        cell.clock,
        descent,
        above,
        contact::THRESHOLD,
    )
    .await?;
    let touch = touch.ok_or_else(|| anyhow::anyhow!("the descent ended without contact"))?;
    let height = planner.tip(&touch.q)?.translation.z - world::TABLE_TOP;
    println!(
        "stopped {:.3} s into the descent, grasp point {:.1} mm above the table, torque change {:.2} N·m (at most {:.2} before)",
        touch.after.as_secs_f64(),
        height * 1e3,
        touch.torque,
        quiet
    );
    anyhow::ensure!(
        (-0.005..0.03).contains(&height),
        "the hand stopped {height:.3} m from the table"
    );
    Ok(())
}

async fn run_recover(assets: &Path) -> anyhow::Result<()> {
    let cell = Cell::new(assets, recover::CUBE_AT, &[recover::POST])?;
    let planner = Planner::new(assets, recover::CUBE_AT, &[recover::POST])?;
    let (_, _, report) = recover::recover(cell.arm, planner, cell.cube, cell.clock).await?;
    let placed = report.placed.translation.vector;
    let off =
        ((placed.x - recover::PLACE[0]).powi(2) + (placed.y - recover::PLACE[1]).powi(2)).sqrt();
    println!(
        "ways tried {}, grasps tried {}, cube {:.1} mm from its place",
        report.ways_tried,
        report.grasps_tried,
        off * 1e3
    );
    anyhow::ensure!(
        report.ways_tried == 2,
        "expected the fallback to the second way"
    );
    anyhow::ensure!(
        report.grasps_tried == 3,
        "expected one blocked plan, one miss, one hold"
    );
    anyhow::ensure!(off < 0.005, "the cube ended {off:.3} m from its place");
    anyhow::ensure!(
        (placed.z - REST).abs() < 0.005,
        "the cube is not on the table"
    );
    Ok(())
}

async fn run_batch(assets: &Path) -> anyhow::Result<()> {
    let first = batch::SPOTS[0];
    let cell = Cell::new(assets, first, &[])?;
    let planner = Planner::new(assets, first, &[])?;
    let (_, _, trials) = batch::batch(
        cell.arm,
        planner,
        cell.cube,
        cell.clock,
        cell.body,
        cell.arm_joints,
    )
    .await?;
    println!("spot            repeat  result");
    for trial in &trials {
        let result = match &trial.result {
            Ok(placement) => format!(
                "{:.1} mm off, on the table: {}, {:.2} s",
                placement.off * 1e3,
                placement.on_table,
                placement.seconds
            ),
            Err(failure) => format!("failed: {failure}"),
        };
        println!(
            "({:.2}, {:+.2})    {}      {result}",
            trial.spot[0], trial.spot[1], trial.repeat
        );
    }
    anyhow::ensure!(
        trials.len() == batch::SPOTS.len() * batch::REPEATS,
        "a trial is missing"
    );
    for trial in &trials {
        let reachable = trial.spot != batch::SPOTS[2];
        anyhow::ensure!(
            trial.succeeded() == reachable,
            "trial from {:?} (repeat {}): {:?}",
            trial.spot,
            trial.repeat,
            trial.result
        );
    }
    Ok(())
}

async fn run_sweep(assets: &Path) -> anyhow::Result<()> {
    let spot = batch::SPOTS[0];
    let cell = Cell::new(assets, spot, &[])?;
    let planner = Planner::new(assets, spot, &[])?;
    let target = planner.reach([spot[0], spot[1], REST + ABOVE], Yaw::ALONG_Y, HOME)?;
    let (_, _, runs) = sweep::sweep(cell.arm, planner, cell.clock, target).await?;
    println!("speed  run  out: timed  tracking   back: timed  tracking");
    for repeats in &runs {
        for (n, run) in repeats.iter().enumerate() {
            println!(
                "{:>5.2}  {n:>3}  {:>8.3} s  {:>6.4} rad  {:>8.3} s  {:>6.4} rad",
                run.speed,
                run.out.planned.as_secs_f64(),
                run.out.tracking,
                run.back.planned.as_secs_f64(),
                run.back.tracking
            );
        }
    }
    anyhow::ensure!(runs.len() == sweep::SPEEDS.len(), "a speed is missing");
    anyhow::ensure!(
        runs.iter().all(|repeats| repeats.len() == sweep::REPEATS),
        "a run is missing"
    );
    let timed: Vec<f64> = runs
        .iter()
        .map(|r| r[0].out.planned.as_secs_f64())
        .collect();
    anyhow::ensure!(
        timed.windows(2).all(|pair| pair[1] < pair[0]),
        "faster runs should take less time: {timed:?}"
    );
    Ok(())
}
