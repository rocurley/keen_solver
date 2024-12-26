use std::{
    fs::File,
    io::{stdin, BufRead},
};

use keen_solver::{Solver, SolversStats};

fn main() {
    let stdin = stdin();
    let mut stats = SolversStats::default();
    let mut i = 0;
    for game_seed in stdin.lock().lines() {
        let game_seed = game_seed.unwrap();
        let mut state = keen_solver::parse_game_id(&game_seed);
        let mut last_state = state.clone();
        while !state.solved() {
            state.try_solvers(Some(&mut stats));
            let last_solver = stats.history.last().unwrap();
            if Solver::RadialSearchPromising == last_solver.solver {
                let file_a = File::create(format!("hard_examples/{:04}a", i)).unwrap();
                last_state.write_save(file_a);
                let file_b = File::create(format!("hard_examples/{:04}b", i)).unwrap();
                state.write_save(file_b);
                i += 1;
            }
            last_state = state.clone();
        }
    }
}
