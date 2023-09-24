use std::io::{stdin, stdout, BufRead};

use keen_solver::{self, GameState};

fn main() {
    let mut state = GameState::from_save(stdin().lock());
    state.filter_by_blocks_simple();
    while state.try_solvers() {}
    state.write_save(stdout());
}
