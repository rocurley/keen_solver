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
    blocks: Vec<Constraint>,
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

    let mut seen = HashMap::new();
    let mut next_block_id = 0;
    let cell_blocks: Vec<CellInfo> = (0..size * size)
        .map(|i| {
            *seen.entry(blocks_uf.find(i)).or_insert_with(|| {
                let block_id = next_block_id;
                next_block_id += 1;
                CellInfo {
                    possibilities: ((2 << size) - 1),
                    block_id,
                }
            })
        })
        .collect();

    /*
    for row in cell_blocks.chunks_exact(size) {
        let s: String = row.iter().map(|c| format!("{:02}", c)).collect();
        println!("{}", s);
    }
    */

    let constraints: Vec<_> = constraints_pair
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
            Constraint { op, val }
        })
        .collect();
    GameState {
        desc,
        size,
        blocks: constraints,
        cells: cell_blocks,
    }
}

impl GameState {
    pub fn write_save(&self, mut out: impl Write) {
        let mut write = |key, value: &str| {
            write!(out, "{}:{}:{}\n", key, value.len(), value).unwrap();
        };

        write("SAVEFILE", "Simon Tatham's Portable Puzzle Collection");
        write("VERSION ", "1");
        write("GAME    ", "Keen");
        write("PARAMS  ", &format!("{}du", self.size));
        write("CPARAMS ", &format!("{}du", self.size));
        write("DESC    ", &self.desc);
        write("NSTATES ", "1");
        write("STATEPOS", "1");
    }
}
