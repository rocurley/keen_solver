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
}

impl BlockInfo {
    fn is_linear(&self, board_size: usize) -> bool {
        let i0 = self.cells[0];
        self.cells.iter().all(|i| i / board_size == i0 / board_size)
            || self.cells.iter().all(|i| i % board_size == i0 % board_size)
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
    fn conditional_possibilities(
        &self,
        possibilities: &[Bitmask],
        board_size: usize,
    ) -> Vec<Bitmask> {
        let mut values = vec![1; self.cells.len()];
        let mut new_possibilities = vec![0; self.cells.len()];
        let mut first = true;
        'outer: while first || next_values_list(board_size, &mut values) {
            first = false;
            // TODO: colinear filter
            for (pos, x) in possibilities.iter().zip(values.iter()) {
                if pos & (1 << (x - 1)) == 0 {
                    continue 'outer;
                }
            }
            for (pos, x) in new_possibilities.iter_mut().zip(values.iter()) {
                *pos |= (1 << (x - 1));
            }
        }
        new_possibilities
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
    // TODO: Does not appear to work. Add test for conditional_possibilities?
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
            .filter(|(i, cell)| Bitmask::is_power_of_two(cell.possibilities))
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
}
