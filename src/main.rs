use std::io::{stdin, stdout, BufRead};

use keen_solver::{self, SolverStats};

fn main() {
    let stdin = stdin();
    let mut stats = SolverStats::default();
    for game_seed in stdin.lock().lines() {
        let game_seed = game_seed.unwrap();
        let mut state = keen_solver::parse_game_id(&game_seed);
        state.filter_by_blocks_simple();
        while !state.solved() && state.try_solvers(Some(&mut stats)) {}
        if state.solved() {
            //eprintln!("");
            continue;
        }
        eprintln!("Got to: {}", game_seed);
        state.write_save(stdout());
        return;
    }
    dbg!(stats);
}
