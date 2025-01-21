use std::{
    cmp,
    collections::HashMap,
    fmt::Debug,
    iter::zip,
    simd::{cmp::SimdPartialEq, LaneCount, Simd, SupportedLaneCount},
};

use bumpalo::Bump;
use pest::Parser;
use pest_derive::Parser;
use union_find::{QuickUnionUf, UnionByRank, UnionFind};

use crate::{
    delete_from_vector,
    factorization::{self, factorize, Factorization},
    permutation::visit_lexical_permutations,
};

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
enum Operator {
    Add,
    Mul,
    Sub,
    Div,
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub struct Constraint {
    op: Operator,
    val: i32,
}

impl Constraint {
    #[cfg(test)]
    fn satisfied_by(self, v: &[i8]) -> bool {
        match self.op {
            Operator::Add => v.iter().map(|x| *x as i32).sum::<i32>() == self.val,
            Operator::Mul => v.iter().map(|x| *x as i32).product::<i32>() == self.val,
            Operator::Sub => {
                v[0] as i32 - v[1] as i32 == self.val || v[1] as i32 - v[0] as i32 == self.val
            }
            Operator::Div => {
                v[0] as i32 * self.val == v[1] as i32 || v[1] as i32 * self.val == v[0] as i32
            }
        }
    }
}

pub type Bitmask = u8;

#[readonly::make]
#[derive(Debug, PartialEq, Eq, Clone)]
pub struct CellInfo {
    pub block_id: Vec<usize>,
    pub possibilities: Vec<Bitmask>,
}

#[readonly::make]
#[derive(Debug, PartialEq, Eq, Clone)]
pub struct BlockInfo<'arena> {
    #[readonly]
    pub constraint: Constraint,
    #[readonly]
    pub cells: Vec<usize>,
    #[readonly]
    pub interacting_blocks: Vec<usize>,
    #[readonly]
    pub possibilities: Vec<&'arena [i8]>,
    pub must_be_in_block_eligible: bool,
    pub compatibility_search_eligible: bool,
    pub radial_search_eligible: bool,
}

impl<'arena> BlockInfo<'arena> {
    fn add_interacting(&mut self, mask: u64) {
        self.interacting_blocks
            .reserve_exact(mask.count_ones() as usize);
        for i in 0..64 {
            if (mask & (1 << i)) > 0 {
                self.interacting_blocks.push(i);
            }
        }
    }
    fn fill_in_possibilities(&mut self, arena: &'arena Bump, board_size: usize) {
        self.possibilities =
            constraint_satisfying_values(arena, self.constraint, &self.cells, board_size);
    }
}

// TODO: maybe return a vec?
fn constraint_satisfying_values<'arena>(
    arena: &'arena Bump,
    constraint: Constraint,
    cells: &[usize],
    size: usize,
) -> Vec<&'arena [i8]> {
    let count = cells.len();
    let ctx = PossibilityContext::new(size, count, cells);
    match constraint.op {
        Operator::Add => ctx.addition_possibilities(arena, constraint.val),
        Operator::Mul => ctx.multiplication_possibilities(arena, constraint.val),
        Operator::Sub => {
            let n = constraint.val as i8;
            (1..=size as i8 - n)
                .flat_map(move |x| {
                    [
                        &*arena.alloc_slice_fill_iter([x, n + x]),
                        &*arena.alloc_slice_fill_iter([n + x, x]),
                    ]
                })
                .filter(|v| ctx.no_conflict(v))
                .collect()
        }
        Operator::Div => {
            let n = constraint.val as i8;
            (1..=size as i8 / n)
                .flat_map(move |x| {
                    [
                        &*arena.alloc_slice_fill_iter([x, n * x]),
                        &*arena.alloc_slice_fill_iter([n * x, x]),
                    ]
                })
                .filter(|v| ctx.no_conflict(v))
                .collect()
        }
    }
}

struct PossibilityContext {
    size: usize,
    count: usize,
    masks: CellMasks,
}

impl PossibilityContext {
    fn new(size: usize, count: usize, cells: &[usize]) -> Self {
        let masks = CellMasks::new(size, cells);
        PossibilityContext { size, count, masks }
    }
    fn no_conflict(&self, possibility: &[i8]) -> bool {
        self.masks.no_conflict(possibility)
    }

    fn addition_possibilities<'arena>(
        &self,
        arena: &'arena Bump,
        target: i32,
    ) -> Vec<&'arena [i8]> {
        let set_to_min = |i: usize, v: &mut Vec<i8>| {
            let remaining = target - v[..i].iter().map(|x| *x as i32).sum::<i32>();
            let min = cmp::max(
                1,
                remaining - self.size as i32 * (self.count - 1 - i) as i32,
            );
            v[i] = min as i8;
        };
        let max = |i: usize, v: &Vec<i8>| -> i8 {
            let remaining = target - v[..i].iter().map(|x| *x as i32).sum::<i32>();
            let max = cmp::min(self.size as i32, remaining - (self.count - 1 - i) as i32);
            max as i8
        };
        let mut out = Vec::new();
        let mut v = vec![0; self.count];
        for i in 0..self.count {
            set_to_min(i, &mut v);
        }
        loop {
            if self.no_conflict(&v) {
                out.push(&*arena.alloc_slice_copy(&v));
            }
            let increment_point = v[..self.count - 1]
                .iter()
                .enumerate()
                .rev()
                .find(|(i, x)| **x < max(*i, &v));
            let Some((i, _)) = increment_point else {
                return out;
            };
            v[i] += 1;
            for i in i + 1..self.count {
                set_to_min(i, &mut v);
            }
        }
    }

    fn multiplication_possibilities<'arena>(
        &self,
        arena: &'arena Bump,
        target: i32,
    ) -> Vec<&'arena [i8]> {
        let target = factorize(target);
        let mut by_rank = vec![Vec::new(); 5];
        let mut out = Vec::new();
        for x in 1..=self.size as i32 {
            let f = factorize(x);
            let slot = &mut by_rank[f.rank()];
            slot.push((x, f));
        }
        let mut v = vec![
            ProductPossibility {
                rank: 4,
                i: 0,
                val: 1,
                factorization: factorization::ONE,
            };
            self.count
        ];
        let reset_range = |start, v: &mut [ProductPossibility]| {
            let mut required = target / v[..start].iter().map(|x| x.factorization).product();
            for i in start..self.count - 1 {
                let rank = required.rank();
                let rank_ix = if i > 0 && v[i - 1].rank == rank {
                    v[i - 1].i
                } else {
                    0
                };
                debug_assert!(
                    rank_ix < by_rank[rank].len(),
                    "by_rank: {:?}\nrequired:{:?}\nrank:{}\nsize:{}",
                    by_rank,
                    required,
                    rank,
                    self.size,
                );
                let (val, factorization) = by_rank[rank][rank_ix];
                v[i] = ProductPossibility {
                    rank,
                    i: rank_ix,
                    val,
                    factorization,
                };
                required /= factorization;
            }
            if !required.is_whole() {
                return false;
            }
            let remainder = required.product();
            let rank = required.rank();
            v[self.count - 1] = ProductPossibility {
                val: remainder,
                rank,
                factorization: required,
                // Garbage value, we won't use it later
                i: 0,
            };
            let prior = v[self.count - 2];
            // Normally we compare by i instead of val, but val is monotonic in i so this works
            // too.
            ((prior.rank, prior.val) <= (rank, remainder))
                && (1..=self.size as i32).contains(&remainder)
        };
        if reset_range(0, &mut v) {
            self.write_permutations(arena, &v, &mut out);
        }
        loop {
            debug_assert!(v[..self.count - 1]
                .iter()
                .all(|x| by_rank[x.rank][x.i].0 == x.val));
            let increment_point = v[..self.count - 1]
                .iter()
                .enumerate()
                .rev()
                .find(|(_, x)| x.i < by_rank[x.rank].len() - 1);
            let Some((i, _)) = increment_point else {
                break;
            };
            v[i].i += 1;
            (v[i].val, v[i].factorization) = by_rank[v[i].rank][v[i].i];
            if !(v[..=i]
                .iter()
                .map(|x| x.factorization)
                .product::<Factorization>()
                .divides(target))
            {
                continue;
            }
            if reset_range(i + 1, &mut v) {
                self.write_permutations(arena, &v, &mut out);
            }
        }
        out
    }

    fn write_permutations<'arena>(
        &self,
        arena: &'arena Bump,
        v: &[ProductPossibility],
        out: &mut Vec<&'arena [i8]>,
    ) {
        let mut v: Vec<i8> = v.iter().map(|x| x.val as i8).collect();
        out.reserve((1..=v.len()).product());
        let visit = |perm: &[i8]| {
            if self.no_conflict(perm) {
                out.push(&*arena.alloc_slice_copy(perm));
            }
        };
        visit_lexical_permutations(&mut v, visit);
    }
}

struct CellMasksInner<const LANES: usize>
where
    LaneCount<LANES>: SupportedLaneCount,
{
    masks: Vec<Simd<u64, LANES>>,
}

impl<const LANES: usize> CellMasksInner<LANES>
where
    LaneCount<LANES>: SupportedLaneCount,
{
    fn new(size: usize, cells: &[usize]) -> Self {
        let masks = cells
            .iter()
            .map(|&ix| {
                let mut mask = Simd::<u64, LANES>::splat(0);
                let i = (ix % size) * size;
                mask.as_mut_array()[i / 64] |= 1 << (i % 64);
                let i = (size + ix / size) * size;
                mask.as_mut_array()[i / 64] |= 1 << (i % 64);
                mask
            })
            .collect();
        CellMasksInner { masks }
    }
    fn no_conflict(&self, v: &[i8]) -> bool {
        let mut seen = Simd::splat(0);
        for (x, mask) in zip(v, &self.masks) {
            let mask = mask << (x - 1) as u64;
            if (seen & mask).simd_ne(Simd::splat(0)).any() {
                return false;
            }
            seen |= mask;
        }
        true
    }
}

enum CellMasks {
    Domino,
    Lanes1(CellMasksInner<1>),
    Lanes2(CellMasksInner<2>),
    Lanes4(CellMasksInner<4>),
}

impl CellMasks {
    fn new(size: usize, cells: &[usize]) -> Self {
        if cells.len() == 2 {
            return Self::Domino;
        }
        match 2 * size * size {
            0..=64 => Self::Lanes1(CellMasksInner::new(size, cells)),
            65..=128 => Self::Lanes2(CellMasksInner::new(size, cells)),
            129..=256 => Self::Lanes4(CellMasksInner::new(size, cells)),
            _ => panic!("Size {} larger than expected", size),
        }
    }
    fn no_conflict(&self, v: &[i8]) -> bool {
        match self {
            CellMasks::Domino => v[0] != v[1],
            CellMasks::Lanes1(inner) => inner.no_conflict(v),
            CellMasks::Lanes2(inner) => inner.no_conflict(v),
            CellMasks::Lanes4(inner) => inner.no_conflict(v),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ProductPossibility {
    rank: usize,
    i: usize,
    val: i32,
    factorization: Factorization,
}

#[derive(Debug, Clone)]
pub struct GameState<'arena> {
    pub desc: String,
    pub size: usize,
    pub cells: CellInfo,
    pub blocks: Vec<BlockInfo<'arena>>,
    pub rows_exclude_n_in_n_eligible: Vec<bool>,
    pub cols_exclude_n_in_n_eligible: Vec<bool>,
    pub rows_only_in_block_eligible: Vec<bool>,
    pub cols_only_in_block_eligible: Vec<bool>,
    pub skip_inelligible: bool,
    col_idx: Simd<usize, 8>,
}

impl PartialEq for GameState<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.desc == other.desc
            && self.size == other.size
            && self.cells == other.cells
            && self.blocks == other.blocks
            && self.rows_exclude_n_in_n_eligible == other.rows_exclude_n_in_n_eligible
            && self.cols_exclude_n_in_n_eligible == other.cols_exclude_n_in_n_eligible
            && self.rows_only_in_block_eligible == other.rows_only_in_block_eligible
            && self.cols_only_in_block_eligible == other.cols_only_in_block_eligible
            && self.skip_inelligible == other.skip_inelligible
            && self.col_idx == other.col_idx
    }
}
impl Eq for GameState<'_> {}

pub struct RowCopy {
    y: usize,
    transposed: bool,
    pub possibilities: Simd<Bitmask, 8>,
}

pub fn index(size: usize, x: usize, y: usize, transposed: bool) -> usize {
    if transposed {
        x * size + y
    } else {
        y * size + x
    }
}

impl<'arena> GameState<'arena> {
    pub fn mask_cell_possibilities(&mut self, cell_id: usize, mask: Bitmask) -> bool {
        let original = self.cells.possibilities[cell_id];
        if original == original & mask {
            return false;
        }
        // Filter block possibilities by the new cell possibilities
        let block_id = self.cells.block_id[cell_id];
        let block = &mut self.blocks[block_id];
        let cell_ix_in_block = block.cells.iter().position(|&ix| ix == cell_id).unwrap();
        block.possibilities.retain(|block_possibility| {
            (1 << (block_possibility[cell_ix_in_block] - 1)) & mask > 0
        });
        self.apply_block_possibilities_to_cells(block_id);
        true
    }
    pub fn board_as_vec(&self) -> Simd<u8, 64> {
        assert!(self.size <= 8);
        Simd::load_or_default(&self.cells.possibilities)
    }
    pub fn update_board(&mut self, new_board: Simd<u8, 64>) -> bool {
        let old_board = self.board_as_vec();
        let changed = old_board.simd_eq(new_board);
        if !changed.any() {
            return false;
        }
        let mut changed_blocks = vec![usize::MAX; self.size * self.size];
        let all_blocks = Simd::load_or_default(&self.cells.block_id);
        all_blocks.store_select(&mut changed_blocks, changed.cast());
        changed_blocks.sort_unstable();
        changed_blocks.dedup();
        changed_blocks.pop(); // Remove dummy usize::MAX value
        for block_id in changed_blocks {
            let block = &mut self.blocks[block_id];
            block.possibilities.retain(|&block_possibility| {
                zip(block_possibility, &block.cells).all(|(&x, &cell_ix)| {
                    (1 << (x - 1)) & self.cells.possibilities[cell_ix] > 0
                })
            });
            self.apply_block_possibilities_to_cells(block_id);
        }
        true
    }
    pub fn get_row(&self, y: usize, transposed: bool) -> RowCopy {
        let possibilities = if transposed {
            Simd::gather_or_default(&self.cells.possibilities[y..], self.col_idx)
        } else {
            Simd::load_or_default(&self.cells.possibilities[self.size * y..self.size * (y + 1)])
        };
        RowCopy {
            y,
            transposed,
            possibilities,
        }
    }
    // TODO: we update blocks multiple times, which doesn't seem efficient.
    pub fn update_row(&mut self, row: RowCopy) -> bool {
        let mut changed = false;
        for x in 0..self.size {
            let ix = index(self.size, x, row.y, row.transposed);
            changed |= self.mask_cell_possibilities(ix, row.possibilities[x]);
        }
        changed
    }
    // NOTE: will incorrectly refresh eligibility if the order is different.
    pub fn replace_block_possibilities(
        &mut self,
        block_id: usize,
        new_possibilities: Vec<&'arena [i8]>,
    ) -> bool {
        let block = &mut self.blocks[block_id];
        if block.possibilities == new_possibilities {
            return false;
        }
        block.possibilities = new_possibilities;
        self.apply_block_possibilities_to_cells(block_id);
        true
    }
    pub fn delete_block_possibilities(
        &mut self,
        block_id: usize,
        delete_indices: impl Iterator<Item = usize>,
    ) -> bool {
        let block = &mut self.blocks[block_id];
        let mut delete_indices = delete_indices.peekable();
        if delete_indices.peek().is_none() {
            return false;
        }
        delete_from_vector(&mut block.possibilities, delete_indices);
        self.apply_block_possibilities_to_cells(block_id);
        true
    }
    fn apply_block_possibilities_to_cells(&mut self, block_id: usize) {
        let block = &mut self.blocks[block_id];
        // Filter block cell possibilities by new block possibilities
        let mut new_cell_values = vec![0; block.cells.len()];
        for block_possibility in &block.possibilities {
            for (i, possibility) in block_possibility.iter().enumerate() {
                new_cell_values[i] |= 1 << (possibility - 1);
            }
        }
        // Update cells and cell-level elgigibility
        for (new_cell_value, &ix) in new_cell_values.into_iter().zip(&block.cells) {
            let cell_value = &mut self.cells.possibilities[ix];
            if *cell_value == new_cell_value {
                continue;
            }
            *cell_value = new_cell_value;
            let x = ix % self.size;
            let y = ix / self.size;
            self.rows_exclude_n_in_n_eligible[y] = true;
            self.cols_exclude_n_in_n_eligible[x] = true;
        }
        self.mark_block_changed(block_id);
    }
    /*
    No solver considers the entire board at once. They consider subsets of the board. For each
    solver, for each iteration,  there's a subset of the board it takes as input, and a subset
    of the board it writes to.
        ExcludeNInN:
            Depends on arbitrary subsets of a row or column, affects the compliment within that
            row or column.
        MustBeInBlock:
            Depends on a block, affects the block's neighbors
        CompatibilitySearch:
            Depends on a neighborhood, affects the center of the neighborhood.
        RadialSearch(Promising):
            Depends on a neighborhood, affects the center of the neighborhood.
    The basic theory, then, is that we should run a solver on a target when the target's
    dependencies for that solver have changed since the last time that solver was run on that
    target. Probably the easiest way to track this is to, for every solver, keep track of what
    items are eligible for that solver. ExcludeNInN would be a problem if you got really fine
    grained with it, but keeping it to the row and column level for now should be fine.
     */
    fn mark_block_changed(&mut self, block_id: usize) {
        let block = &mut self.blocks[block_id];
        block.must_be_in_block_eligible = true;
        block.compatibility_search_eligible = true;
        block.radial_search_eligible = true;
        for &ix in &block.cells {
            let x = ix % self.size;
            let y = ix / self.size;
            self.rows_only_in_block_eligible[y] = true;
            self.cols_only_in_block_eligible[x] = true;
        }
        // Use indices to avoid holding a mutable reference to self.blocks
        for i in 0..block.interacting_blocks.len() {
            let neighbor_id = self.blocks[block_id].interacting_blocks[i];
            let neighbor = &mut self.blocks[neighbor_id];
            neighbor.compatibility_search_eligible = true;
            neighbor.radial_search_eligible = true;
        }
    }
    fn initialize_cell_possibilities(&mut self) {
        for block in &self.blocks {
            for &possibility in &block.possibilities {
                for (&value, &i) in zip(possibility, &block.cells) {
                    self.cells.possibilities[i] |= 1 << (value - 1);
                }
            }
        }
    }
    pub fn from_save(arena: &'arena Bump, r: impl std::io::BufRead) -> Self {
        let mut kvs = HashMap::new();
        for line in r.lines() {
            let line = line.unwrap();
            let mut split = line.split(':');
            let key = split.next().unwrap().trim().to_owned();
            split.next().unwrap();
            let val = split.next().unwrap().to_owned();
            kvs.entry(key).or_insert_with(Vec::new).push(val);
        }
        let params = &kvs["PARAMS"][0];
        let size: usize = params[..params.len() - 2].parse().unwrap();
        let desc = &kvs["DESC"][0];
        let game_id = format!("{}:{}", size, desc);
        let mut out = parse_game_id(arena, &game_id);
        let mut cell_possibilities = vec![0; size * size];
        for raw_move in kvs["MOVE"].iter() {
            let mut split = raw_move[1..].split(',');
            let x: usize = split.next().unwrap().parse().unwrap();
            let y: usize = split.next().unwrap().parse().unwrap();
            let v: usize = split.next().unwrap().parse().unwrap();
            let i = x + y * size;
            cell_possibilities[i] |= 1 << (v - 1);
        }
        for (i, mask) in cell_possibilities.into_iter().enumerate() {
            out.mask_cell_possibilities(i, mask);
        }
        out
    }
    pub fn print_save(&self) {
        let mut scratch_vec = Vec::new();
        self.write_save(&mut scratch_vec);
        let scratch = String::from_utf8(scratch_vec).unwrap();
        println!("{}", scratch);
    }
    pub fn consistency_check(&self) {
        for (i, &possibilities) in self.cells.possibilities.iter().enumerate() {
            if possibilities == 0 {
                self.print_save();
                panic!(
                    "Cell with no possibilities at {}, {}",
                    i % self.size,
                    i / self.size
                );
            }
        }
    }
}

#[derive(Parser)]
#[grammar = "game_id.pest"]
struct GameIDParser;

pub fn parse_game_id<'arena>(arena: &'arena Bump, raw: &'_ str) -> GameState<'arena> {
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
                must_be_in_block_eligible: true,
                compatibility_search_eligible: true,
                radial_search_eligible: true,
            }
        })
        .collect();

    let mut seen = HashMap::new();
    let mut next_block_id = 0;
    let block_id: Vec<_> = (0..size * size)
        .map(|i| {
            let block_id = *seen.entry(blocks_uf.find(i)).or_insert_with(|| {
                let block_id = next_block_id;
                next_block_id += 1;
                block_id
            });
            blocks[block_id].cells.push(i);
            block_id
        })
        .collect();
    let cells = CellInfo {
        block_id,
        possibilities: vec![0; size * size],
    };
    for (block_id, block) in blocks.iter_mut().enumerate() {
        let mut mask = 0;
        for &i in &block.cells {
            let x = i % size;
            let y = i / size;
            for x2 in 0..size {
                let other_block_id = cells.block_id[x2 + y * size];
                if other_block_id != block_id {
                    mask |= 1 << other_block_id;
                }
            }
            for y2 in 0..size {
                let other_block_id = cells.block_id[x + y2 * size];
                if other_block_id != block_id {
                    mask |= 1 << other_block_id;
                }
            }
        }
        block.add_interacting(mask);
    }
    for block in blocks.iter_mut() {
        block.fill_in_possibilities(arena, size);
    }
    let col_index_vec: Vec<_> = (0..size).map(|x| x * size).collect();
    let col_idx = Simd::load_or(&col_index_vec, Simd::splat(usize::MAX));
    let mut out = GameState {
        desc,
        size,
        blocks,
        cells,
        rows_exclude_n_in_n_eligible: vec![true; size],
        cols_exclude_n_in_n_eligible: vec![true; size],
        rows_only_in_block_eligible: vec![true; size],
        cols_only_in_block_eligible: vec![true; size],
        skip_inelligible: true,
        col_idx,
    };
    out.initialize_cell_possibilities();
    out
}

#[cfg(test)]
mod tests {

    use std::{cmp::max, collections::HashSet, iter::zip};

    use super::{constraint_satisfying_values, CellMasks, Constraint, Operator};
    use bumpalo::Bump;
    use prop::collection::vec;
    use proptest::prelude::*;

    fn next_values_list(board_size: usize, xs: &mut [i8]) -> bool {
        let board_size = board_size as i8;
        for x in xs {
            if *x < board_size {
                *x += 1;
                return true;
            }
            *x = 1;
        }
        false
    }

    fn constraint_satisfying_values_brute(
        constraint: Constraint,
        count: usize,
        size: usize,
    ) -> Vec<Vec<i8>> {
        let mut scratch = vec![1; count];
        let mut out = Vec::new();
        while next_values_list(size, &mut scratch) {
            if !constraint.satisfied_by(&scratch) {
                continue;
            }
            out.push(scratch.to_vec());
        }
        out
    }

    fn any_op() -> impl Strategy<Value = Operator> {
        use Operator::*;
        prop_oneof![Just(Add), Just(Mul), Just(Sub), Just(Div),].no_shrink()
    }

    #[derive(Clone, Debug)]
    struct TestCase {
        size: usize,
        count: usize,
        constraint: Constraint,
    }

    fn count(op: Operator) -> BoxedStrategy<usize> {
        use Operator::*;
        match op {
            Sub | Div => Strategy::boxed(Just(2usize)),
            Add | Mul => Strategy::boxed(2usize..6),
        }
    }
    fn val(op: Operator, size: usize, count: usize) -> BoxedStrategy<i32> {
        use Operator::*;
        match op {
            Add => Strategy::boxed(count as i32..(count * size) as i32),
            Mul => Strategy::boxed(
                vec(1..=size as i32, count).prop_map(|v| v.into_iter().product()),
            ),
            Sub => Strategy::boxed(1..(size as i32 - 1)),
            Div => Strategy::boxed(2..size as i32),
        }
    }

    fn test_case() -> impl Strategy<Value = TestCase> {
        (4usize..9, any_op()).prop_flat_map(|(size, op)| {
            count(op).prop_flat_map(move |count| {
                val(op, size, count).prop_map(move |val| TestCase {
                    size,
                    count,
                    constraint: Constraint { val, op },
                })
            })
        })
    }

    proptest! {
        #[test]
        fn test_constraint_satisfying_values(tc in test_case()) {
            let mut expected = constraint_satisfying_values_brute(tc.constraint, tc.count, tc.size);
            prop_assume!(!expected.is_empty());
            // We can't disable conflict checking for count = 2, so filter by count here.
            if tc.count == 2 {
                expected.retain(|v| v[0] != v[1]);
            }
            let arena = Bump::new();
            // For count > 2, we don't want to test internal conflict checking, so we make cells
            // that don't conflict.
            prop_assume!(tc.size >= tc.count);
            let cells : Vec<_> = (0..tc.count).map(|i| i * (tc.size + 1)).collect();
            let mut actual :Vec<_>= constraint_satisfying_values(&arena, tc.constraint, &cells, tc.size);
            expected.sort();
            actual.sort();
            assert_eq!(expected, actual);
        }
    }
    #[derive(Copy, Clone, Debug)]
    enum Direction {
        Up,
        Down,
        Left,
        Right,
    }
    impl Direction {
        fn any() -> impl Strategy<Value = Direction> {
            prop_oneof![
                Just(Direction::Up),
                Just(Direction::Down),
                Just(Direction::Left),
                Just(Direction::Right),
            ]
        }
    }
    fn block_from_directions(directions: Vec<Direction>) -> (usize, Vec<usize>) {
        let mut coords = vec![(0, 0)];
        for d in directions {
            let (x0, y0) = *coords.last().unwrap();
            let new = match d {
                Direction::Up => (x0, y0 - 1),
                Direction::Down => (x0, y0 + 1),
                Direction::Left => (x0 - 1, y0),
                Direction::Right => (x0 + 1, y0),
            };
            coords.push(new);
        }
        let xmin = coords.iter().map(|(x, _)| *x).min().unwrap();
        let xmax = coords.iter().map(|(x, _)| *x).max().unwrap();
        let ymin = coords.iter().map(|(_, y)| *y).min().unwrap();
        let ymax = coords.iter().map(|(_, y)| *y).max().unwrap();
        let size = max(xmax - xmin, ymax - ymin) as usize + 1;
        let mut ixs: Vec<_> = coords
            .into_iter()
            .map(|(x, y)| ((x - xmin) + size as i32 * (y - ymin)) as usize)
            .collect();
        ixs.sort_unstable();
        ixs.dedup();
        (size, ixs)
    }
    fn block_strategy() -> impl Strategy<Value = (usize, Vec<usize>, Vec<i8>)> {
        vec(Direction::any(), 1..8).prop_flat_map(|directions| {
            let (size, cells) = block_from_directions(directions);
            vec(1..=size as i8, cells.len())
                .prop_map(move |values| (size, cells.clone(), values))
        })
    }

    fn no_conflict_simple(size: usize, cells: &[usize], elements: &[i8]) -> bool {
        let mut rows = HashSet::new();
        let mut cols = HashSet::new();
        for (&i, &val) in zip(cells, elements) {
            let x = i % size;
            let y = i / size;
            if !rows.insert((y, val)) || !cols.insert((x, val)) {
                return false;
            }
        }
        true
    }

    proptest! {
        #[test]
        fn test_cell_masks((size, cells, elements) in block_strategy()) {
            let expected = no_conflict_simple(size, &cells, &elements);
            let masks = CellMasks::new(size, &cells);
            let actual = masks.no_conflict(&elements);
            assert_eq!(expected, actual);
        }
    }
}
