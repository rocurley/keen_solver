use std::io::stdout;

use keen_solver;

fn main() {
    let mut state = keen_solver::parse_game_id(
        "6:__b_3aa__b_aa_a3b_3a_4aa__a_6a_a_,a5s1a8s2a7m4s2a12m6d2s1m6m4a8m24d2s2",
    );
    state.filter_by_block_possibilities();
    state.write_save(stdout());
}
