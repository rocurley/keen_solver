use std::{
    cmp,
    collections::HashMap,
    fmt::Debug,
    iter::{empty, once, zip},
    simd::Simd,
    usize,
};

use bumpalo::Bump;
use pest::Parser;
use pest_derive::Parser;
use union_find::{QuickUnionUf, UnionByRank, UnionFind};

use crate::delete_from_vector;

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
    fn add_interacting(&mut self, ix: usize) {
        if !self.interacting_blocks.contains(&ix) {
            self.interacting_blocks.push(ix);
        }
    }
    fn fill_in_possibilities(&mut self, arena: &'arena Bump, board_size: usize) {
        self.possibilities =
            constraint_satisfying_values(arena, self.constraint, &self.cells, board_size);
    }
}

// TODO: maybe return a vec?
fn constraint_satisfying_values<'arena, 'a>(
    arena: &'arena Bump,
    constraint: Constraint,
    cells: &'a [usize],
    size: usize,
) -> Vec<&'arena [i8]> {
    let count = cells.len();
    let ctx = PossibilityContext { size, count, cells };
    match constraint.op {
        Operator::Add => ctx.addition_possibilities(arena, constraint.val),
        Operator::Mul => multiplication_possibilities_bad(size, count, count, constraint.val)
            .filter(|v| ctx.no_conflict(v))
            .map(|v| &*arena.alloc_slice_copy(&v))
            .collect(),
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

fn multiplication_possibilities_bad(
    size: usize,
    count: usize,
    remaining_count: usize,
    target: i32,
) -> Box<dyn Iterator<Item = Vec<i8>>> {
    if (size as i32).pow(remaining_count as u32) < target {
        return Box::new(empty());
    }
    if remaining_count == 1 {
        return Box::new(once(vec![target as i8]));
    }
    let divisors = (1..=size as i32).filter(move |x| target % x == 0);
    let out = divisors.flat_map(move |x| {
        let new_target = target / x;
        multiplication_possibilities_bad(size, count, remaining_count - 1, new_target).map(
            move |mut xs| {
                xs.push(x as i8);
                xs
            },
        )
    });
    Box::new(out)
}

struct PossibilityContext<'a> {
    size: usize,
    count: usize,
    cells: &'a [usize],
}

impl<'a> PossibilityContext<'a> {
    fn no_conflict(&self, possibility: &[i8]) -> bool {
        for ((i, x), ci) in possibility.iter().enumerate().zip(self.cells.iter()) {
            for (y, cj) in possibility[i + 1..].iter().zip(self.cells[i + 1..].iter()) {
                if x == y
                    && (ci / self.size == cj / self.size || ci % self.size == cj % self.size)
                {
                    return false;
                }
            }
        }
        true
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

impl<'arena> PartialEq for GameState<'arena> {
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
impl<'arena> Eq for GameState<'arena> {}

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
        for neighbor_id in block.interacting_blocks.clone() {
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
        possibilities: vec![(2 << size) - 1; size * size],
    };
    for (i, &block_id) in cells.block_id.iter().enumerate() {
        let x = i % size;
        let y = i / size;
        for x2 in 0..size {
            let other_block_id = cells.block_id[x2 + y * size];
            if other_block_id != block_id {
                blocks[block_id].add_interacting(other_block_id);
            }
        }
        for y2 in 0..size {
            let other_block_id = cells.block_id[x + y2 * size];
            if other_block_id != block_id {
                blocks[block_id].add_interacting(other_block_id);
            }
        }
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
    use super::{constraint_satisfying_values, Constraint, Operator};
    use bumpalo::Bump;
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
        prop_oneof![Just(Add), Just(Mul), Just(Sub), Just(Div),]
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
    fn val(op: Operator, size: usize) -> BoxedStrategy<i32> {
        use Operator::*;
        match op {
            Add | Mul => Strategy::boxed(1i32..500),
            Sub => Strategy::boxed(1..(size as i32 - 1)),
            Div => Strategy::boxed(2..size as i32),
        }
    }
    prop_compose! {
        fn test_case()(size in 4usize..9, op in any_op())
            (count in count(op), op in Just(op), size in Just(size),val in val(op, size)) -> TestCase {
            TestCase{size, count, constraint: Constraint{val, op}}
        }
    }

    proptest! {
        #[test]
        fn test_constraint_satisfying_values(tc in test_case()) {
            let mut expected = constraint_satisfying_values_brute(tc.constraint, tc.count, tc.size);
            prop_assume!(!expected.is_empty());
            let arena = Bump::new();
            // We don't want to test internal conflict checking, so we make cells that don't
            // conflict.
            prop_assume!(tc.size >= tc.count);
            let cells : Vec<_> = (0..tc.count).map(|i| i * (tc.size + 1)).collect();
            let mut actual :Vec<_>= constraint_satisfying_values(&arena, tc.constraint, &cells, tc.size);
            expected.sort();
            actual.sort();
            assert_eq!(expected, actual);
        }
    }
}
