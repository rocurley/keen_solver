use std::io::{stdin, stdout, BufRead};

use keen_solver;

fn main() {
    let stdin = stdin();
    for game_seed in stdin.lock().lines() {
        let game_seed = game_seed.unwrap();
        let mut state = keen_solver::parse_game_id(&game_seed);
        state.filter_by_blocks_simple();
        while state.try_solvers() {}
        if state.solved() {
            continue;
        }
        eprintln!("Got to: {}", game_seed);
        state.write_save(stdout());
        return;
    }
}
