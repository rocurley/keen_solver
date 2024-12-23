use std::io::{stdin, stdout};

use keen_solver::game::GameState;

fn main() {
    let mut state = GameState::from_save(stdin().lock());
    state.try_solvers(None);
    state.write_save(stdout());
}
