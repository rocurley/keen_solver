use std::io::{stdin, stdout, BufRead};

use keen_solver;

fn main() {
    let stdin = stdin();
    for game_seed in stdin.lock().lines() {
        let mut state = keen_solver::parse_game_id(&game_seed.unwrap());
        state.filter_by_blocks_simple();
        for _ in 0..100 {
            state.exclude_n_in_n();
            state.filter_by_blocks_conditional();
            state.compatibility_search();
        }
        if state.solved() {
            continue;
        }
        state.write_save(stdout());
        return;
    }
}
