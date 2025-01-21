#![feature(portable_simd)]

mod bitset;
mod compatibility_search;
mod exclude_n_in_n;
mod factorization;
pub mod game;
mod must_be_in_block;
mod only_in_block;
mod permutation;
mod radial_search;

use std::{
    fmt::{Debug, Display},
    io::Write,
    mem::swap,
    ops::{Index, IndexMut},
    simd::{LaneCount, Simd, SimdElement, SupportedLaneCount},
    time::{Duration, Instant},
};

pub use game::parse_game_id;
use game::{Bitmask, GameState};
use tabled::{settings::Style, Table, Tabled};

fn iterate_possibilities(size: usize, possiblities: Bitmask) -> impl Iterator<Item = usize> {
    (0..size)
        .filter(move |i| possiblities & (1 << i) > 0)
        .map(|i| i + 1)
}

fn load_with_default<T: SimdElement, const LANES: usize>(xs: &[T], default: T) -> Simd<T, LANES>
where
    LaneCount<LANES>: SupportedLaneCount,
{
    let default = Simd::splat(default);
    Simd::load_or(xs, default)
}

fn delete_from_vector<T>(xs: &mut Vec<T>, mut iter: impl Iterator<Item = usize>) {
    let Some(mut dst) = iter.next() else {
        return;
    };
    let mut src = dst + 1;
    for i in iter.chain([xs.len()]) {
        while src < i {
            let (front, back) = xs.split_at_mut(src);
            swap(&mut front[dst], &mut back[0]);
            dst += 1;
            src += 1;
        }
        src += 1;
    }
    xs.truncate(dst);
}

impl GameState<'_> {
    pub fn write_save(&self, mut out: impl Write) {
        let mut write = |key, value: &str| {
            writeln!(out, "{}:{}:{}", key, value.len(), value).unwrap();
        };
        let pencil_moves: Vec<_> = self
            .cells
            .possibilities
            .iter()
            .enumerate()
            .flat_map(|(i, &possibilities)| {
                iterate_possibilities(self.size, possibilities).map(move |x| (i, x))
            })
            .collect();
        let definite_moves: Vec<_> = self
            .cells
            .possibilities
            .iter()
            .enumerate()
            .filter(|(_, &possibilities)| Bitmask::is_power_of_two(possibilities))
            .flat_map(|(i, &possibilities)| {
                iterate_possibilities(self.size, possibilities).map(move |x| (i, x))
            })
            .collect();

        let n_moves = pencil_moves.len() + definite_moves.len() + 1;

        write("SAVEFILE", "Simon Tatham's Portable Puzzle Collection");
        write("VERSION ", "1");
        write("GAME    ", "Keen");
        write("PARAMS  ", &format!("{}du", self.size));
        write("CPARAMS ", &format!("{}du", self.size));
        write("DESC    ", &self.desc);
        write("NSTATES ", &format!("{}", n_moves));
        write("STATEPOS", &format!("{}", n_moves));
        for (i, x) in pencil_moves {
            write(
                "MOVE    ",
                &format!("P{},{},{}", i % self.size, i / self.size, x),
            );
        }
        for (i, x) in definite_moves {
            write(
                "MOVE    ",
                &format!("R{},{},{}", i % self.size, i / self.size, x),
            );
        }
    }
    pub fn solved(&self) -> bool {
        self.cells
            .possibilities
            .iter()
            .all(|&possibilties| Bitmask::is_power_of_two(possibilties))
    }

    pub fn try_solvers(&mut self, mut stats: Option<&mut SolversStats>) -> bool {
        if self.run_solver(Solver::ExcludeNInN, &mut stats) {
            return true;
        }
        if self.run_solver(Solver::MustBeInBlock, &mut stats) {
            return true;
        }
        if self.run_solver(Solver::OnlyInBlock, &mut stats) {
            return true;
        }
        if self.run_solver(Solver::CompatibilitySearch, &mut stats) {
            return true;
        }
        if self.run_solver(Solver::RadialSearchPromising, &mut stats) {
            return true;
        }
        if self.run_solver(Solver::RadialSearch, &mut stats) {
            return true;
        }
        false
    }
    fn run_solver(&mut self, solver: Solver, stats: &mut Option<&mut SolversStats>) -> bool {
        let initial_entropy = self.entropy();
        let start = Instant::now();
        let res = match solver {
            Solver::ExcludeNInN => self.exclude_n_in_n(),
            Solver::MustBeInBlock => self.must_be_in_block(),
            Solver::OnlyInBlock => self.only_in_block(),
            Solver::CompatibilitySearch => self.compatibility_search(),
            Solver::RadialSearchPromising => self.radial_search_promising(),
            Solver::RadialSearch => self.radial_search(),
        };
        dbg!(solver, res);
        if let Some(ref mut stats) = stats {
            let entropy_removed = if res {
                initial_entropy - self.entropy()
            } else {
                0.0
            };
            let log_entry = SolverLogEntry {
                solver,
                success: res,
                entropy_removed,
                duration: start.elapsed(),
            };
            stats.log(log_entry);
        }
        res
    }
    fn entropy(&self) -> f32 {
        self.blocks
            .iter()
            .map(|b| f32::log2(b.possibilities.len() as f32))
            .sum()
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Solver {
    ExcludeNInN,
    MustBeInBlock,
    OnlyInBlock,
    CompatibilitySearch,
    RadialSearchPromising,
    RadialSearch,
}
impl Tabled for Solver {
    const LENGTH: usize = 1;

    fn fields(&self) -> Vec<std::borrow::Cow<'_, str>> {
        vec![format!("{:?}", self).into()]
    }

    fn headers() -> Vec<std::borrow::Cow<'static, str>> {
        vec!["solver".into()]
    }
}

impl Display for Solver {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Debug::fmt(self, f)
    }
}

#[derive(Clone, Debug, Default, Tabled)]
pub struct SolverStats {
    calls: usize,
    successes: usize,
    entropy_removed: f32,
    #[tabled(display_with = "display_duration")]
    duration: Duration,
}

#[derive(Clone, Debug, Tabled)]
pub struct SolverLogEntry {
    pub solver: Solver,
    pub success: bool,
    pub entropy_removed: f32,
    #[tabled(display_with = "display_duration")]
    pub duration: Duration,
}

fn display_duration(d: &Duration) -> String {
    format!("{:?}", d)
}

#[derive(Clone, Debug, Default)]
pub struct SolversStats {
    pub exclude_n_in_n: SolverStats,
    pub must_be_in_block: SolverStats,
    pub only_in_block: SolverStats,
    pub compatibility_search: SolverStats,
    pub radial_search_promising: SolverStats,
    pub radial_search: SolverStats,
    pub history: Vec<SolverLogEntry>,
}

impl Index<Solver> for SolversStats {
    type Output = SolverStats;
    fn index(&self, index: Solver) -> &Self::Output {
        match index {
            Solver::ExcludeNInN => &self.exclude_n_in_n,
            Solver::MustBeInBlock => &self.must_be_in_block,
            Solver::OnlyInBlock => &self.only_in_block,
            Solver::CompatibilitySearch => &self.compatibility_search,
            Solver::RadialSearchPromising => &self.radial_search_promising,
            Solver::RadialSearch => &self.radial_search,
        }
    }
}
impl IndexMut<Solver> for SolversStats {
    fn index_mut(&mut self, index: Solver) -> &mut Self::Output {
        match index {
            Solver::ExcludeNInN => &mut self.exclude_n_in_n,
            Solver::MustBeInBlock => &mut self.must_be_in_block,
            Solver::OnlyInBlock => &mut self.only_in_block,
            Solver::CompatibilitySearch => &mut self.compatibility_search,
            Solver::RadialSearchPromising => &mut self.radial_search_promising,
            Solver::RadialSearch => &mut self.radial_search,
        }
    }
}

impl SolversStats {
    fn log(&mut self, entry: SolverLogEntry) {
        let agg = &mut self[entry.solver];
        agg.calls += 1;
        if entry.success {
            agg.successes += 1;
            agg.entropy_removed += entry.entropy_removed;
            agg.duration += entry.duration;
        }
        self.history.push(entry);
    }
    pub fn show_stats(&self) {
        use Solver::*;
        let solvers = [
            ExcludeNInN,
            MustBeInBlock,
            OnlyInBlock,
            CompatibilitySearch,
            RadialSearchPromising,
            RadialSearch,
        ];
        let table_data = solvers.into_iter().map(|solver| {
            (
                solver,
                &self[solver],
                self[solver].entropy_removed / self[solver].duration.as_secs_f32() / 1000.0,
            )
        });

        let mut tab = Table::new(table_data);
        tab.with(Style::sharp());
        println!("{}", tab);
    }
    pub fn show_trace(&self) {
        let mut tab = Table::new(&self.history);
        tab.with(Style::sharp());
        println!("{}", tab);
    }
}

#[cfg(test)]
mod tests {
    use crate::{delete_from_vector, GameState};
    use bumpalo::Bump;
    use prop::collection::vec;
    use proptest::prelude::*;
    #[test]
    #[ignore]
    fn test_load() {
        let save = std::fs::read("test_data/search_depth_2_test_case").unwrap();
        let arena = Bump::new();
        let gs = GameState::from_save(&arena, save.as_slice());
        let mut new_save = Vec::new();
        gs.write_save(&mut new_save);
        assert_eq!(save, new_save);
    }
    proptest! {
        #[test]
        fn test_delete_from_vector(
            mut v in vec(any::<u8>(), 0..100),
            mut to_delete in vec(0usize..100, 0..100),
        ) {
            to_delete.sort();
            to_delete.dedup();
            to_delete.retain(|i| *i < v.len());
            let mut expected = v.clone();
            for i in to_delete.iter().rev() {
                expected.remove(*i);
            }
            delete_from_vector(&mut v, to_delete.into_iter());
            assert_eq!(expected, v);
        }
    }
}
