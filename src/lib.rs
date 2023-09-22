use std::{collections::HashMap, io::Write};

use pest::Parser;
use pest_derive::Parser;
use union_find::{QuickUnionUf, UnionByRank, UnionFind};

// 6:__b_3aa__b_aa_a3b_3a_4aa__a_6a_a_,a5s1a8s2a7m4s2a12m6d2s1m6m4a8m24d2s2
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
}

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
        &self,
        possibilities: &'a [Bitmask],
        board_size: usize,
    ) -> JointPossibilities<'a> {
        let mut values = vec![1; self.cells.len()];
        values[0] = 0;
        JointPossibilities {
            constraint: self.constraint,
            possibilities,
            board_size,
            values,
        }
    }
    fn conditional_possibilities(
        &self,
        possibilities: &[Bitmask],
        board_size: usize,
    ) -> Vec<Bitmask> {
        let mut new_possibilities = vec![0; self.cells.len()];
        let mut iter = self.joint_possibilities(possibilities, board_size);
        while let Some(values) = iter.next() {
            for (pos, x) in new_possibilities.iter_mut().zip(values.iter()) {
                *pos |= 1 << (x - 1);
            }
        }
        new_possibilities
    }
}

struct JointPossibilities<'a> {
    constraint: Constraint,
    possibilities: &'a [Bitmask],
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
    GameState {
        desc,
        size,
        blocks,
        cells,
    }
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
    pub fn exclude_n_in_n(&mut self) {
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
                                if (1 << x) & cell_mask > 0 {
                                    self.cells[ix].possibilities &= seen;
                                } else {
                                    self.cells[ix].possibilities &= !seen;
                                }
                            }
                        }
                        Ordering::Greater => {}
                    }
                }
            }
        }
    }
    pub fn filter_by_blocks_conditional(&mut self) {
        for block in &self.blocks {
            let possibilities: Vec<Bitmask> = block
                .cells
                .iter()
                .map(|&i| self.cells[i].possibilities)
                .collect();
            let new_possibilities = block.conditional_possibilities(&possibilities, self.size);
            for (&i, p) in block.cells.iter().zip(new_possibilities.into_iter()) {
                self.cells[i].possibilities = p;
            }
        }
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
    pub fn get_block_possibilities(&self, i: usize) -> Vec<Bitmask> {
        self.blocks[i]
            .cells
            .iter()
            .map(|&i| self.cells[i].possibilities)
            .collect()
    }
    pub fn set_block_possibilities(&mut self, i: usize, new_masks: &[Bitmask]) {
        for (&i, &new_mask) in self.blocks[i].cells.iter().zip(new_masks.iter()) {
            self.cells[i].possibilities = new_mask;
        }
    }
    // Depth 1 for now
    pub fn compatibility_search(&mut self) {
        for block_id in 0..self.blocks.len() {
            let block = &self.blocks[block_id];
            let masks = self.get_block_possibilities(block_id);
            let mut new_masks = vec![0; block.cells.len()];
            let mut iter = block.joint_possibilities(&masks, self.size);
            while let Some(p) = iter.next() {
                let neighbors_compatible = block.interacting_blocks.iter().all(|&neighbor_id| {
                    let neighbor = &self.blocks[neighbor_id];
                    let neighbor_masks = self.get_block_possibilities(neighbor_id);
                    let mut neighbor_iter =
                        neighbor.joint_possibilities(&neighbor_masks, self.size);
                    while let Some(np) = neighbor_iter.next() {
                        if joint_possibilities_compatible(
                            self.size,
                            p,
                            &block.cells,
                            np,
                            &neighbor.cells,
                        ) {
                            return true;
                        }
                    }
                    false
                });
                if neighbors_compatible {
                    for (pos, x) in new_masks.iter_mut().zip(p.iter()) {
                        *pos |= 1 << (x - 1);
                    }
                }
            }
            self.set_block_possibilities(block_id, &new_masks);
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
    use crate::{BlockInfo, Constraint};

    #[test]
    fn test_conditional_possibilities() {
        let c = Constraint {
            op: crate::Operator::Div,
            val: 2,
        };
        let b = BlockInfo {
            constraint: c,
            cells: vec![0, 1],
            interacting_blocks: Vec::new(),
        };
        let possibilities = vec![1 << (4 - 1), 1 << (2 - 1) | 1 << (3 - 1) | 1 << (6 - 1)];
        let result = b.conditional_possibilities(&possibilities, 6);
        assert_eq!(result, vec![1 << (4 - 1), 1 << (2 - 1)]);
    }
}
