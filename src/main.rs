use std::io::stdout;

use keen_solver;

const easy_game: &str =
    "6:__aa_3a_3a_3aa__aa_ba3__aa_aa_4a_3a,m30d2a5d3s1m10a5m5s1m6a12d3s1s1a6d2s1";
const hard_game: &str = "6:__b_3aa__b_aa_a3b_3a_4aa__a_6a_a_,a5s1a8s2a7m4s2a12m6d2s1m6m4a8m24d2s2";

fn main() {
    let mut state = keen_solver::parse_game_id(easy_game);
    state.filter_by_blocks_simple();
    state.exclude_n_in_n();
    state.filter_by_blocks_conditional();
    state.write_save(stdout());
}
