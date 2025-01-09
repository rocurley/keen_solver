use std::io::{stdin, stdout};

use bumpalo::Bump;
use keen_solver::game::GameState;

fn main() {
    let arena = Bump::new();
    let mut state = GameState::from_save(&arena, stdin().lock());
    state.try_solvers(None);
    state.write_save(stdout());
}
