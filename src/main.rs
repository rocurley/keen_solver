use std::io::{stdin, stdout, BufRead};

use keen_solver;

fn main() {
    let stdin = stdin();
    for game_seed in stdin.lock().lines() {
        let mut state = keen_solver::parse_game_id(&game_seed.unwrap());
        state.filter_by_blocks_simple();
        'outer: loop {
            if state.exclude_n_in_n() {
                continue;
            }
            if state.filter_by_blocks_conditional() {
                continue;
            }
            for depth in 0..4 {
                if state.compatibility_search(depth) {
                    continue 'outer;
                }
            }
            break;
        }
        if state.solved() {
            continue;
        }
        state.write_save(stdout());
        return;
    }
}
