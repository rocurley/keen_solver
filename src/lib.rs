use std::{
    collections::HashMap,
    io::Write,
    ops::{Index, IndexMut},
};

use pest::Parser;
use pest_derive::Parser;
use union_find::{QuickUnionUf, UnionByRank, UnionFind};

#[derive(Parser)]
#[grammar = "game_id.pest"]
struct GameIDParser;

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
enum Operator {
    Add,
    Mul,
    Sub,
    Div,
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
struct Constraint {
    op: Operator,
    val: i32,
}

impl Constraint {
    fn satisfied_by(self, v: &[i32]) -> bool {
        match self.op {
            Operator::Add => v.iter().sum::<i32>() == self.val,
            Operator::Mul => v.iter().product::<i32>() == self.val,
            Operator::Sub => v[0] - v[1] == self.val || v[1] - v[0] == self.val,
            Operator::Div => {
                (v[0] % v[1] == 0 && v[0] / v[1] == self.val)
                    || (v[1] % v[0] == 0 && v[1] / v[0] == self.val)
            }
        }
    }
}

#[derive(Debug, PartialEq, Eq, Clone)]
struct BlockInfo {
    constraint: Constraint,
    cells: Vec<usize>,
    interacting_blocks: Vec<usize>,
    possibilities: Vec<Vec<i32>>,
}

const SEARCH_DEPTH: usize = 5;

impl BlockInfo {
    fn is_linear(&self, board_size: usize) -> bool {
        let i0 = self.cells[0];
        self.cells.iter().all(|i| i / board_size == i0 / board_size)
            || self.cells.iter().all(|i| i % board_size == i0 % board_size)
    }
    fn add_interacting(&mut self, ix: usize) {
        if !self.interacting_blocks.contains(&ix) {
            self.interacting_blocks.push(ix);
        }
    }
    fn possibilities<'a>(&'a self, board_size: usize) -> impl 'a + Iterator<Item = Vec<i32>> {
        let is_linear = self.is_linear(board_size);
        let mut it: Box<dyn Iterator<Item = (i32, Vec<i32>)>> =
            Box::new((1..(board_size as i32) + 1).map(|x| (x, vec![x])));
        for _ in 1..self.cells.len() {
            it = Box::new(it.flat_map(move |(mut lb, v)| {
                if is_linear {
                    lb += 1;
                }
                (lb..(board_size as i32) + 1).map(move |x| {
                    let mut v2 = v.clone();
                    v2.push(x);
                    (x, v2)
                })
            }));
        }
        it.map(|(_, v)| v)
            .filter(|v| self.constraint.satisfied_by(v))
    }
    fn joint_possibilities<'a>(
        &'a self,
        possibilities: &'a [Bitmask],
        board_size: usize,
    ) -> JointPossibilities<'a> {
        let mut values = vec![1; self.cells.len()];
        values[0] = 0;
        JointPossibilities {
            constraint: self.constraint,
            cells: &self.cells,
            possibilities,
            board_size,
            values,
        }
    }
    fn fill_in_possibilities(&mut self, board_size: usize) {
        let dummy_cell_possibilities = vec![(1 << board_size) - 1; self.cells.len()];
        let mut iter = self.joint_possibilities(&dummy_cell_possibilities, board_size);
        let mut possiblities = Vec::new();
        while let Some(p) = iter.next() {
            possiblities.push(p.to_vec());
        }
        self.possibilities = possiblities;
    }
}

struct JointPossibilities<'a> {
    constraint: Constraint,
    possibilities: &'a [Bitmask],
    cells: &'a [usize],
    board_size: usize,
    values: Vec<i32>,
}

impl<'a> JointPossibilities<'a> {
    pub fn next(&mut self) -> Option<&[i32]> {
        'outer: while next_values_list(self.board_size, &mut self.values) {
            for (pos, x) in self.possibilities.iter().zip(self.values.iter()) {
                if pos & (1 << (x - 1)) == 0 {
                    continue 'outer;
                }
            }
            if !self.constraint.satisfied_by(&self.values) {
                continue;
            }
            for ((i, x), ci) in self.values.iter().enumerate().zip(self.cells.iter()) {
                for (y, cj) in self.values[i + 1..].iter().zip(self.cells[i + 1..].iter()) {
                    if x == y {
                        if ci / self.board_size == cj / self.board_size
                            || ci % self.board_size == cj % self.board_size
                        {
                            continue 'outer;
                        }
                    }
                }
            }
            return Some(&self.values);
        }
        None
    }
}

fn next_values_list(board_size: usize, xs: &mut [i32]) -> bool {
    let board_size = board_size as i32;
    for x in xs {
        if *x < board_size {
            *x += 1;
            return true;
        }
        *x = 1;
    }
    false
}

type Bitmask = u8;

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
struct CellInfo {
    block_id: usize,
    possibilities: Bitmask,
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct GameState {
    desc: String,
    size: usize,
    cells: Vec<CellInfo>,
    blocks: Vec<BlockInfo>,
}

pub fn parse_game_id(raw: &str) -> GameState {
    let parsed = GameIDParser::parse(Rule::game_id, raw)
        .unwrap()
        .next()
        .unwrap();
    let mut parsed_iter = parsed.into_inner();
    let size_pair = parsed_iter.next().unwrap();
    let gaps_pair = parsed_iter.next().unwrap();
    let constraints_pair = parsed_iter.next().unwrap();

    let size: usize = size_pair.as_str().parse().unwrap();
    let desc = format!("{},{}", gaps_pair.as_str(), constraints_pair.as_str());

    let mut lines = vec![false; 2 * size * (size - 1) + 1];
    let mut i = 0;
    for gap in gaps_pair.into_inner() {
        let mut gap_iter = gap.into_inner();
        let letter_num = gap_iter.next().unwrap().as_str();
        let gap_size = match letter_num.chars().last().unwrap() {
            '_' => 0,
            c => (c as usize - 'a' as usize) + 25 * (letter_num.len() - 1) + 1,
        };
        let repeat_count: usize = match gap_iter.next().unwrap().as_str() {
            "" => 1,
            s => s.parse().unwrap(),
        };
        for _ in 0..repeat_count {
            i += gap_size;
            lines[i] = true;
            i += 1;
        }
    }
    //let lines_str: String = lines.iter().map(|b| if *b { '|' } else { '_' }).collect();
    //println!("{}", lines_str);
    //println!("||__||||_|_|||__||_|_||_|_|_|__||||_|||||_|_|||_|||||||_||_|");

    let mut blocks_uf = QuickUnionUf::<UnionByRank>::new(size * size);
    let mut lines_row_cols = lines.chunks_exact(size - 1);
    for y in 0..size {
        let row_lines = lines_row_cols.next().unwrap();
        for (x, line) in row_lines.iter().enumerate() {
            if !line {
                blocks_uf.union(y * size + x, y * size + x + 1);
            }
        }
    }
    for x in 0..size {
        let col_lines = lines_row_cols.next().unwrap();
        for (y, line) in col_lines.iter().enumerate() {
            if !line {
                blocks_uf.union(y * size + x, (y + 1) * size + x);
            }
        }
    }

    let mut blocks: Vec<_> = constraints_pair
        .into_inner()
        .map(|pair| {
            let (op_str, val_str) = pair.as_str().split_at(1);
            let op = match op_str {
                "a" => Operator::Add,
                "m" => Operator::Mul,
                "s" => Operator::Sub,
                "d" => Operator::Div,
                _ => panic!("unknown operator"),
            };
            let val = val_str.parse().unwrap();
            BlockInfo {
                constraint: Constraint { op, val },
                cells: Vec::new(),
                interacting_blocks: Vec::new(),
                possibilities: Vec::new(),
            }
        })
        .collect();

    let mut seen = HashMap::new();
    let mut next_block_id = 0;
    let cells: Vec<CellInfo> = (0..size * size)
        .map(|i| {
            let block_id = *seen.entry(blocks_uf.find(i)).or_insert_with(|| {
                let block_id = next_block_id;
                next_block_id += 1;
                block_id
            });
            blocks[block_id].cells.push(i);
            CellInfo {
                possibilities: ((2 << size) - 1),
                block_id,
            }
        })
        .collect();
    for (i, cell) in cells.iter().enumerate() {
        let block_id = cell.block_id;
        let x = i % size;
        let y = i / size;
        for x2 in 0..size {
            let other_block_id = cells[x2 + y * size].block_id;
            if other_block_id != block_id {
                blocks[block_id].add_interacting(other_block_id);
            }
        }
        for y2 in 0..size {
            let other_block_id = cells[x + y2 * size].block_id;
            if other_block_id != block_id {
                blocks[block_id].add_interacting(other_block_id);
            }
        }
    }
    for block in blocks.iter_mut() {
        block.fill_in_possibilities(size);
    }
    let mut out = GameState {
        desc,
        size,
        blocks,
        cells,
    };
    out.cells_from_blocks();
    out
}

fn iterate_possibilities(size: usize, possiblities: Bitmask) -> impl Iterator<Item = usize> {
    (0..size)
        .filter(move |i| possiblities & (1 << i) > 0)
        .map(|i| i + 1)
}

fn index(size: usize, x: usize, y: usize, transposed: bool) -> usize {
    if transposed {
        x * size + y
    } else {
        y * size + x
    }
}

impl GameState {
    fn cells_from_blocks(&mut self) {
        for block in &self.blocks {
            for (i, &cell_id) in block.cells.iter().enumerate() {
                let mut mask = 0;
                for possibility in &block.possibilities {
                    mask |= 1 << (possibility[i] - 1);
                }
                self.cells[cell_id].possibilities &= mask;
            }
        }
    }
    pub fn blocks_from_cells(&mut self) {
        for block in &mut self.blocks {
            block.possibilities.retain(|p| {
                p.iter().zip(block.cells.iter()).all(|(x, &cell_id)| {
                    self.cells[cell_id].possibilities & (1 << (x - 1)) != 0
                })
            });
        }
    }
    pub fn filter_by_blocks_simple(&mut self) {
        let block_possibilities: Vec<Bitmask> = self
            .blocks
            .iter()
            .map(|b| {
                let mut mask = 0;
                for possibility in b.possibilities(self.size) {
                    for x in possibility {
                        mask |= 1 << (x - 1);
                    }
                }
                mask
            })
            .collect();
        for cell in self.cells.iter_mut() {
            cell.possibilities &= block_possibilities[cell.block_id];
        }
    }
    pub fn exclude_n_in_n(&mut self) -> bool {
        let mut made_progress = false;
        for transposed in [true, false] {
            for y in 0..self.size {
                for cell_mask in 1..1 << self.size {
                    let mut seen = 0;
                    for x in 0..self.size {
                        if (1 << x) & cell_mask > 0 {
                            let ix = index(self.size, x, y, transposed);
                            seen |= self.cells[ix].possibilities;
                        }
                    }
                    use std::cmp::Ordering;
                    match Bitmask::count_ones(seen).cmp(&Bitmask::count_ones(cell_mask)) {
                        Ordering::Less => {
                            panic!("fewer possibilities than cells: kill this branch");
                        }
                        Ordering::Equal => {
                            for x in 0..self.size {
                                let ix = index(self.size, x, y, transposed);
                                let old = self.cells[ix].possibilities;
                                if (1 << x) & cell_mask > 0 {
                                    self.cells[ix].possibilities &= seen;
                                } else {
                                    self.cells[ix].possibilities &= !seen;
                                }
                                made_progress |= old != self.cells[ix].possibilities;
                            }
                        }
                        Ordering::Greater => {}
                    }
                }
            }
        }
        if made_progress {
            self.blocks_from_cells();
            self.cells_from_blocks();
        }
        made_progress
    }
    pub fn write_save(&self, mut out: impl Write) {
        let mut write = |key, value: &str| {
            write!(out, "{}:{}:{}\n", key, value.len(), value).unwrap();
        };
        let pencil_moves: Vec<_> = self
            .cells
            .iter()
            .enumerate()
            .flat_map(|(i, cell)| {
                iterate_possibilities(self.size, cell.possibilities).map(move |x| (i, x))
            })
            .collect();
        let definite_moves: Vec<_> = self
            .cells
            .iter()
            .enumerate()
            .filter(|(_, cell)| Bitmask::is_power_of_two(cell.possibilities))
            .flat_map(|(i, cell)| {
                iterate_possibilities(self.size, cell.possibilities).map(move |x| (i, x))
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
            .iter()
            .all(|cell| Bitmask::is_power_of_two(cell.possibilities))
    }
    pub fn set_block_possibilities(&mut self, i: usize, new_masks: &[Bitmask]) -> bool {
        let mut changed = false;
        for (&i, &new_mask) in self.blocks[i].cells.iter().zip(new_masks.iter()) {
            changed |= self.cells[i].possibilities != new_mask;
            self.cells[i].possibilities = new_mask;
        }
        changed
    }
    pub fn compatibility_search(&mut self, depth: usize) -> bool {
        let mut made_progress = false;
        for block_id in 0..self.blocks.len() {
            made_progress |= self.compatibility_search_single(block_id, depth);
        }
        if made_progress {
            self.cells_from_blocks();
        }
        made_progress
    }
    fn compatibility_search_single(&mut self, block_id: usize, depth: usize) -> bool {
        let block = &self.blocks[block_id];
        let mut seen = vec![None; self.size * self.size];
        let old_joint_possibilities = &block.possibilities;
        // TODO: abort search when down to one possibility
        let new_joint_possibilities: Vec<_> = old_joint_possibilities
            .iter()
            .filter(|p| self.compatibility_search_inner(depth, block_id, p, &mut seen))
            .cloned()
            .collect();
        if new_joint_possibilities != *old_joint_possibilities {
            self.blocks[block_id].possibilities = new_joint_possibilities;
            true
        } else {
            false
        }
    }

    pub fn radial_search(&mut self, depth: usize) -> bool {
        assert_eq!(depth, 1);
        let mut made_progress = false;
        for block_id in 0..self.blocks.len() {
            let new_possibilities: Vec<_> = self.blocks[block_id]
                .possibilities
                .iter()
                .enumerate()
                .filter(|&(i, _)| self.radial_search_single(block_id, i))
                .map(|(_, p)| p.clone())
                .collect();
            if new_possibilities != self.blocks[block_id].possibilities {
                made_progress = true;
                self.blocks[block_id].possibilities = new_possibilities;
            }
        }
        if made_progress {
            self.cells_from_blocks();
        }
        made_progress
    }

    // depth 1 for now
    fn radial_search_single(&self, block_id: usize, possibility_ix: usize) -> bool {
        let block = &self.blocks[block_id];
        struct SearchBlock<'a> {
            block: &'a BlockInfo,
            possibility_ix: usize,
            // TODO: cache which cells this block's cells can see
        }
        impl<'a> SearchBlock<'a> {
            fn can_increment(&self) -> bool {
                self.possibility_ix + 1 < self.block.possibilities.len()
            }
        }
        let mut search_space = vec![SearchBlock {
            block,
            possibility_ix,
        }];
        search_space.extend(block.interacting_blocks.iter().map(|&i| SearchBlock {
            block: &self.blocks[i],
            possibility_ix: 0,
        }));
        loop {
            let validation_failure = (1..search_space.len()).find(|&i| {
                let r_block = &search_space[i];
                search_space[..i].iter().any(|l_block| {
                    !joint_possibilities_compatible(
                        self.size,
                        &l_block.block.possibilities[l_block.possibility_ix],
                        &l_block.block.cells,
                        &r_block.block.possibilities[r_block.possibility_ix],
                        &r_block.block.cells,
                    )
                })
            });
            let mut increment_point = match validation_failure {
                None => return true,
                Some(x) => x,
            };
            while increment_point > 0 && !search_space[increment_point].can_increment() {
                increment_point -= 1;
            }
            if increment_point == 0 {
                return false;
            }
            search_space[increment_point].possibility_ix += 1;
            for block in &mut search_space[increment_point + 1..] {
                block.possibility_ix = 0;
            }
        }
    }

    // checks if there's any solution consistent with seen
    fn compatibility_search_inner(
        &self,
        depth: usize,
        block_id: usize,
        block_joint_possibilities: &[i32],
        seen: &mut [Option<Vec<i32>>],
    ) -> bool {
        if depth == 0 {
            return true;
        }
        let block = &self.blocks[block_id];
        seen[block_id] = Some(block_joint_possibilities.to_vec());
        let res = block.interacting_blocks.iter().all(|&neighbor_id| {
            let neighbor = &self.blocks[neighbor_id];
            if let Some(neighbor_joint_possiblities) = seen[neighbor_id].as_ref() {
                return joint_possibilities_compatible(
                    self.size,
                    &block_joint_possibilities,
                    &block.cells,
                    neighbor_joint_possiblities,
                    &neighbor.cells,
                );
            }
            self.blocks[neighbor_id]
                .possibilities
                .iter()
                .any(|neighbor_joint_possiblities| {
                    joint_possibilities_compatible(
                        self.size,
                        &block_joint_possibilities,
                        &block.cells,
                        neighbor_joint_possiblities,
                        &neighbor.cells,
                    ) && self.compatibility_search_inner(
                        depth - 1,
                        neighbor_id,
                        neighbor_joint_possiblities,
                        seen,
                    )
                })
        });
        seen[block_id] = None;
        res
    }
    pub fn must_be_in_block(&mut self) -> bool {
        let mut made_progress = false;
        for (block_id, block) in self.blocks.iter().enumerate() {
            let mut row_required = vec![(1 << self.size) - 1; self.size];
            let mut col_required = vec![(1 << self.size) - 1; self.size];
            for joint_possibilities in &block.possibilities {
                let mut row_vals = vec![0; self.size];
                let mut col_vals = vec![0; self.size];
                for (i, x) in block.cells.iter().zip(joint_possibilities.iter()) {
                    row_vals[i / self.size] |= 1 << x - 1;
                    col_vals[i % self.size] |= 1 << x - 1;
                }
                for (m1, m2) in row_required.iter_mut().zip(row_vals.into_iter()) {
                    *m1 &= m2;
                }
                for (m1, m2) in col_required.iter_mut().zip(col_vals.into_iter()) {
                    *m1 &= m2;
                }
            }
            for (i, cell) in self.cells.iter_mut().enumerate() {
                if cell.block_id == block_id {
                    continue;
                }
                let original = cell.possibilities;
                cell.possibilities &= !row_required[i / self.size];
                cell.possibilities &= !col_required[i % self.size];
                made_progress |= original != cell.possibilities;
            }
        }
        if made_progress {
            self.blocks_from_cells();
            self.cells_from_blocks();
        }
        made_progress
    }
    pub fn try_solvers(&mut self, mut stats: Option<&mut SolverStats>) -> bool {
        if self.run_solver(Solver::ExcludeNInN, &mut stats) {
            return true;
        }
        if self.run_solver(Solver::MustBeInBlock, &mut stats) {
            return true;
        }
        if self.run_solver(Solver::CompatibilitySearch(1), &mut stats) {
            return true;
        }
        if self.run_solver(Solver::RadialSearch(1), &mut stats) {
            return true;
        }
        if self.run_solver(Solver::CompatibilitySearch(3), &mut stats) {
            return true;
        }
        if self.run_solver(Solver::CompatibilitySearch(4), &mut stats) {
            return true;
        }
        return false;
    }
    fn most_interesting_block(&self) -> (usize, &BlockInfo) {
        self.blocks
            .iter()
            .enumerate()
            .filter(|(_, block)| block.possibilities.len() > 1)
            .min_by_key(|(_, block)| {
                (
                    block.cells.len() * 4 * 3 * 5 / block.possibilities.len(),
                    block.cells.len(),
                )
            })
            .unwrap()
    }
    pub fn from_save(r: impl std::io::BufRead) -> Self {
        let mut kvs = HashMap::new();
        for line in r.lines() {
            let line = line.unwrap();
            let mut split = line.split(":");
            let key = split.next().unwrap().trim().to_owned();
            split.next().unwrap();
            let val = split.next().unwrap().to_owned();
            kvs.entry(key).or_insert_with(Vec::new).push(val);
        }
        let params = &kvs["PARAMS"][0];
        let size: usize = params[..params.len() - 2].parse().unwrap();
        let desc = &kvs["DESC"][0];
        let game_id = format!("{}:{}", size, desc);
        let mut out = parse_game_id(&game_id);
        for cell in out.cells.iter_mut() {
            cell.possibilities = 0;
        }
        for raw_move in kvs["MOVE"].iter() {
            let mut split = raw_move[1..].split(",");
            let x: usize = split.next().unwrap().parse().unwrap();
            let y: usize = split.next().unwrap().parse().unwrap();
            let v: usize = split.next().unwrap().parse().unwrap();
            let i = x + y * size;
            out.cells[i].possibilities |= 1 << v - 1;
        }
        out
    }
    fn run_solver(&mut self, solver: Solver, stats: &mut Option<&mut SolverStats>) -> bool {
        let res = match solver {
            Solver::ExcludeNInN => self.exclude_n_in_n(),
            Solver::MustBeInBlock => self.must_be_in_block(),
            Solver::CompatibilitySearch(n) => self.compatibility_search(n),
            Solver::RadialSearch(n) => self.radial_search(n),
        };
        if let Some(ref mut stats) = stats {
            stats[solver].calls += 1;
            if res {
                stats[solver].successes += 1;
            }
        }
        res
    }
}

#[derive(Copy, Clone, Debug)]
enum Solver {
    ExcludeNInN,
    MustBeInBlock,
    CompatibilitySearch(usize),
    RadialSearch(usize),
}

#[derive(Clone, Debug, Default)]
pub struct SolverStat {
    calls: usize,
    successes: usize,
}

#[derive(Clone, Debug, Default)]
pub struct SolverStats {
    exclude_n_in_n: SolverStat,
    must_be_in_block: SolverStat,
    compatibility_search: [SolverStat; SEARCH_DEPTH],
    radial_search: [SolverStat; SEARCH_DEPTH],
}

impl Index<Solver> for SolverStats {
    type Output = SolverStat;
    fn index(&self, index: Solver) -> &Self::Output {
        match index {
            Solver::ExcludeNInN => &self.exclude_n_in_n,
            Solver::MustBeInBlock => &self.must_be_in_block,
            Solver::CompatibilitySearch(n) => &self.compatibility_search[n],
            Solver::RadialSearch(n) => &self.radial_search[n],
        }
    }
}
impl IndexMut<Solver> for SolverStats {
    fn index_mut(&mut self, index: Solver) -> &mut Self::Output {
        match index {
            Solver::ExcludeNInN => &mut self.exclude_n_in_n,
            Solver::MustBeInBlock => &mut self.must_be_in_block,
            Solver::CompatibilitySearch(n) => &mut self.compatibility_search[n],
            Solver::RadialSearch(n) => &mut self.radial_search[n],
        }
    }
}

fn joint_possibilities_compatible(
    size: usize,
    l_values: &[i32],
    l_cells: &[usize],
    r_values: &[i32],
    r_cells: &[usize],
) -> bool {
    for (x, i) in l_values.iter().zip(l_cells.iter()) {
        for (y, j) in r_values.iter().zip(r_cells.iter()) {
            if x == y && (i % size == j % size || i / size == j / size) {
                return false;
            }
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use crate::GameState;
    #[test]
    fn test_load() {
        let save = std::fs::read("test_data/search_depth_2_test_case").unwrap();
        let gs = GameState::from_save(save.as_slice());
        let mut new_save = Vec::new();
        gs.write_save(&mut new_save);
        assert_eq!(save, new_save);
    }
    #[test]
    fn test_depth_2_search() {
        let save = std::fs::read("test_data/search_depth_2_test_case").unwrap();
        let mut gs = GameState::from_save(save.as_slice());
        gs.compatibility_search(3);
        assert_eq!(1 << 3 - 1, gs.cells[5].possibilities);
    }
}
