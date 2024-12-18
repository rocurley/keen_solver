use std::io::{stdin, stdout, BufRead};

use clap::Parser;
use keen_solver::SolverStats;

/// Solve keen puzzles
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// Collect and print statistics
    #[arg(long)]
    stats: bool,
}

fn main() {
    let args = Args::parse();
    let stdin = stdin();
    let mut stats = if args.stats {
        Some(SolverStats::default())
    } else {
        None
    };
    for game_seed in stdin.lock().lines() {
        let game_seed = game_seed.unwrap();
        let mut state = keen_solver::parse_game_id(&game_seed);
        state.filter_by_blocks_simple();
        while !state.solved() && state.try_solvers(stats.as_mut()) {}
        if state.solved() {
            //eprintln!("");
            continue;
        }
        eprintln!("Got to: {}", game_seed);
        state.write_save(stdout());
        return;
    }
    if let Some(stats) = stats {
        dbg!(stats);
    }
}
