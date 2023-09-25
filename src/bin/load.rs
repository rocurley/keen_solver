use std::io::{stdin, stdout};

use keen_solver::{self, GameState};

fn main() {
    let mut state = GameState::from_save(stdin().lock());
    state.filter_by_blocks_simple();
    state.try_solvers(None);
    state.write_save(stdout());
}
