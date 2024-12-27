/*
No solver considers the entire board at once. They consider subsets of the board. For each solver,
for each iteration,  there's a subset of the board it takes as input, and a subset of the board it
writes to.
    ExcludeNInN:
        Depends on arbitrary subsets of a row or column, affects the compliment within that row or column.
    MustBeInBlock:
        Depends on a block, affects the block's neighbors
    CompatibilitySearch:
        Depends on a neighborhood, affects the center of the neighborhood.
    RadialSearch(Promising):
        Depends on a neighborhood, affects the center of the neighborhood.
The basic theory, then, is that we should run a solver on a target when the target's dependencies
for that solver have changed since the last time that solver was run on that target. Probably the
easiest way to track this is to, for every solver, keep track of what items are eligible for that
solver. ExcludeNInN would be a problem if you got really fine grained with it, but keeping it to
the row and column level for now should be fine.
 */

use std::{
    collections::HashSet,
    fmt::{Debug, Display},
    io::Write,
    iter::zip,
    ops::{BitOr, Index, IndexMut},
    time::{Duration, Instant},
};

pub use game::parse_game_id;
use game::{Bitmask, BlockInfo, GameState};
use tabled::{settings::Style, Table, Tabled};

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

pub mod game {
    use std::{collections::HashMap, fmt::Debug};

    use pest::Parser;
    use pest_derive::Parser;
    use union_find::{QuickUnionUf, UnionByRank, UnionFind};

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
        fn satisfied_by(self, v: &[i32]) -> bool {
            match self.op {
                Operator::Add => v.iter().sum::<i32>() == self.val,
                Operator::Mul => v.iter().product::<i32>() == self.val,
                Operator::Sub => v[0] - v[1] == self.val || v[1] - v[0] == self.val,
                Operator::Div => v[0] * self.val == v[1] || v[1] * self.val == v[0],
            }
        }
    }

    pub type Bitmask = u8;

    #[readonly::make]
    #[derive(Debug, PartialEq, Eq, Clone, Copy)]
    pub struct CellInfo {
        pub block_id: usize,
        pub possibilities: Bitmask,
    }

    #[readonly::make]
    #[derive(Debug, PartialEq, Eq, Clone)]
    pub struct BlockInfo {
        #[readonly]
        pub constraint: Constraint,
        #[readonly]
        pub cells: Vec<usize>,
        #[readonly]
        pub interacting_blocks: Vec<usize>,
        #[readonly]
        pub possibilities: Vec<Vec<i32>>,
        pub must_be_in_block_eligible: bool,
        pub compatibility_search_eligible: bool,
        pub radial_search_eligible: bool,
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
        pub fn possibilities(&self, board_size: usize) -> impl '_ + Iterator<Item = Vec<i32>> {
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

    impl JointPossibilities<'_> {
        fn next(&mut self) -> Option<&[i32]> {
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
                        if x == y
                            && (ci / self.board_size == cj / self.board_size
                                || ci % self.board_size == cj % self.board_size)
                        {
                            continue 'outer;
                        }
                    }
                }
                return Some(&self.values);
            }
            None
        }
    }

    #[derive(Debug, PartialEq, Eq, Clone)]
    pub struct GameState {
        pub desc: String,
        pub size: usize,
        pub cells: Vec<CellInfo>,
        pub blocks: Vec<BlockInfo>,
        pub rows_exclude_n_in_n_eligible: Vec<bool>,
        pub cols_exclude_n_in_n_eligible: Vec<bool>,
        pub skip_inelligible: bool,
    }
    impl GameState {
        pub fn mask_cell_possibilities(&mut self, cell_id: usize, mask: Bitmask) -> bool {
            let original = self.cells[cell_id].possibilities;
            if original == original & mask {
                return false;
            }
            // Filter block possibilities by the new cell possibilities
            let block_id = self.cells[cell_id].block_id;
            let block = &mut self.blocks[block_id];
            let cell_ix_in_block = block.cells.iter().position(|&ix| ix == cell_id).unwrap();
            block.possibilities.retain(|block_possibility| {
                (1 << (block_possibility[cell_ix_in_block] - 1)) & mask > 0
            });
            self.apply_block_possibilities_to_cells(block_id);
            true
        }
        // NOTE: will incorrectly refresh eligibility if the order is different.
        pub fn replace_block_possibilities(
            &mut self,
            block_id: usize,
            new_possibilities: Vec<Vec<i32>>,
        ) -> bool {
            let block = &mut self.blocks[block_id];
            if block.possibilities == new_possibilities {
                return false;
            }
            block.possibilities = new_possibilities;
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
                let cell_value = &mut self.cells[ix].possibilities;
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
            for neighbor_id in block.interacting_blocks.clone() {
                let neighbor = &mut self.blocks[neighbor_id];
                neighbor.compatibility_search_eligible = true;
                neighbor.radial_search_eligible = true;
            }
        }
        fn initialize_cell_possibilities(&mut self) {
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
        pub fn from_save(r: impl std::io::BufRead) -> Self {
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
            let mut out = parse_game_id(&game_id);
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
            for (i, cell) in self.cells.iter().enumerate() {
                if cell.possibilities == 0 {
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
                    must_be_in_block_eligible: true,
                    compatibility_search_eligible: true,
                    radial_search_eligible: true,
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
            rows_exclude_n_in_n_eligible: vec![true; size],
            cols_exclude_n_in_n_eligible: vec![true; size],
            skip_inelligible: true,
        };
        out.initialize_cell_possibilities();
        out
    }
}

impl GameState {
    fn exclude_n_in_n_eligible(&mut self, y: usize, transposed: bool) -> &mut bool {
        if transposed {
            &mut self.cols_exclude_n_in_n_eligible[y]
        } else {
            &mut self.rows_exclude_n_in_n_eligible[y]
        }
    }
    fn exclude_n_in_n(&mut self) -> bool {
        let mut made_progress = false;
        let skip_inelligible = self.skip_inelligible;
        for transposed in [true, false] {
            for y in 0..self.size {
                let eligibility = self.exclude_n_in_n_eligible(y, transposed);
                let was_eligible = *eligibility;
                if !*eligibility && skip_inelligible {
                    continue;
                }
                *eligibility = false;
                for cell_mask in 1..1 << self.size {
                    // Union of possibilities in the cells of cell_mask
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
                            self.print_save();
                            dbg!(transposed, y);
                            eprintln!("mask: {:#06b}", cell_mask);
                            panic!("fewer possibilities than cells");
                        }
                        Ordering::Equal => {
                            for x in 0..self.size {
                                let ix = index(self.size, x, y, transposed);
                                if (1 << x) & cell_mask == 0 {
                                    let cell_changed = self.mask_cell_possibilities(ix, !seen);
                                    if cell_changed {
                                        if !was_eligible {
                                            self.print_save();
                                            dbg!(transposed, y);
                                            eprintln!("mask: {:b}", cell_mask);
                                            panic!(
                                                "Supposedly ineligbile row/col made progress."
                                            );
                                        }
                                        made_progress = true;
                                    }
                                }
                            }
                        }
                        Ordering::Greater => {}
                    }
                }
            }
        }
        made_progress
    }
    pub fn write_save(&self, mut out: impl Write) {
        let mut write = |key, value: &str| {
            writeln!(out, "{}:{}:{}", key, value.len(), value).unwrap();
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
    pub fn compatibility_search(&mut self) -> bool {
        let mut made_progress = false;
        for block_id in 0..self.blocks.len() {
            let eligible = self.blocks[block_id].compatibility_search_eligible;
            if !eligible && self.skip_inelligible {
                continue;
            }
            if self.compatibility_search_single(block_id) {
                made_progress = true;
                assert!(
                    eligible,
                    "Supposedly ineligible compatibility search block made progress",
                );
            }
        }
        made_progress
    }
    fn compatibility_search_single(&mut self, block_id: usize) -> bool {
        let block = &self.blocks[block_id];
        let old_joint_possibilities = &block.possibilities;
        // TODO: abort search when down to one possibility
        let new_joint_possibilities: Vec<_> = old_joint_possibilities
            .iter()
            .filter(|p| self.compatibility_search_inner(block_id, p))
            .cloned()
            .collect();
        let made_progress = self.replace_block_possibilities(block_id, new_joint_possibilities);
        self.blocks[block_id].compatibility_search_eligible = false;
        made_progress
    }

    fn radial_search(&mut self) -> bool {
        let mut made_progress = false;
        for block_id in 0..self.blocks.len() {
            let eligible = self.blocks[block_id].radial_search_eligible;
            if !eligible && self.skip_inelligible {
                continue;
            }
            if self.radial_search_single(block_id) {
                made_progress = true;
                assert!(
                    eligible,
                    "Supposedly ineligible radial search block made progress",
                );
            }
        }
        made_progress
    }

    fn radial_search_promising(&mut self) -> bool {
        let (block_id, _) = self.most_interesting_block();
        self.radial_search_single(block_id)
    }

    // radial_search_single filters for consistency with the immediate neighborhood of a block:
    // all blocks that can interact directly with it. For every possibility of the target block, it
    // checks that there exists a consistent sub-solution for the entire neighborhood.
    fn radial_search_single(&mut self, block_id: usize) -> bool {
        let mut new_possibilities = Vec::new();
        let possibilities = &self.blocks[block_id].possibilities;
        for (i, p) in possibilities.iter().enumerate() {
            // We assume that the puzzle is valid, so we can just keep the last possibility if
            // we've eliminated all others.
            if i == possibilities.len() - 1 && new_possibilities.is_empty() {
                new_possibilities.push(p.clone());
                continue;
            }
            if !self.radial_search_single_possibility(block_id, i) {
                continue;
            }
            new_possibilities.push(p.clone());
        }
        let made_progress = self.replace_block_possibilities(block_id, new_possibilities);
        self.blocks[block_id].radial_search_eligible = false;
        made_progress
    }

    fn radial_search_single_possibility(&self, block_id: usize, possibility_ix: usize) -> bool {
        let block = &self.blocks[block_id];
        struct SearchBlock<'a> {
            block: &'a BlockInfo,
            possibility_ix: usize,
            interactions: Vec<CellInteraction>,
        }
        struct CellInteraction {
            local_cell_ix: usize,
            other_block_ix: usize,
            other_cell_ix: usize,
        }
        impl SearchBlock<'_> {
            fn can_increment(&self) -> bool {
                self.possibility_ix + 1 < self.block.possibilities.len()
            }
            fn current_possibility(&self) -> &[i32] {
                &self.block.possibilities[self.possibility_ix]
            }
        }
        let mut search_space = vec![SearchBlock {
            block,
            possibility_ix,
            interactions: Vec::new(),
        }];
        // Setup: store all blocks that can interact with our target block in search_space. For
        // each block, store what earlier blocks they interact with.
        for &i in &block.interacting_blocks {
            let block = &self.blocks[i];
            let mut interactions = Vec::new();
            for (other_block_ix, other_block) in search_space.iter().enumerate() {
                let other_cells = &other_block.block.cells;
                for (other_cell_ix, other_cell_loc) in other_cells.iter().enumerate() {
                    for (local_cell_ix, local_cell_loc) in block.cells.iter().enumerate() {
                        if (other_cell_loc % self.size != local_cell_loc % self.size)
                            && (other_cell_loc / self.size != local_cell_loc / self.size)
                        {
                            continue;
                        }
                        interactions.push(CellInteraction {
                            local_cell_ix,
                            other_block_ix,
                            other_cell_ix,
                        });
                    }
                }
            }
            let sb = SearchBlock {
                block,
                possibility_ix: 0,
                interactions,
            };
            search_space.push(sb);
        }
        loop {
            let validation_failure = (1..search_space.len()).find(|&i| {
                let r_block = &search_space[i];
                r_block.interactions.iter().any(|interaction| {
                    let local_val = r_block.current_possibility()[interaction.local_cell_ix];
                    let other_val = search_space[interaction.other_block_ix]
                        .current_possibility()[interaction.other_cell_ix];
                    local_val == other_val
                })
            });
            let mut increment_point = match validation_failure {
                None => return true,
                Some(x) => x,
            };
            // There's a conflict between search_space[increment_point] and some block in
            // search_space[..increment_point].
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

    // checks if every neighbor has a possibility compatiblle with the given possibility.
    fn compatibility_search_inner(
        &self,
        block_id: usize,
        block_joint_possibility: &[i32],
    ) -> bool {
        let block = &self.blocks[block_id];
        let res = block.interacting_blocks.iter().all(|&neighbor_id| {
            let neighbor = &self.blocks[neighbor_id];
            neighbor
                .possibilities
                .iter()
                .any(|neighbor_joint_possiblity| {
                    joint_possibilities_compatible(
                        self.size,
                        block_joint_possibility,
                        &block.cells,
                        neighbor_joint_possiblity,
                        &neighbor.cells,
                    )
                })
        });
        res
    }
    // If a given block requires that some number be in that block in a specific row or column,
    // filter out that number from the rest of the row or column.
    // Example: filter out 2 from a row containing a /2 block with possibilities (1,2,4)
    fn must_be_in_block(&mut self) -> bool {
        let mut made_progress = false;
        for block_id in 0..self.blocks.len() {
            let block = &self.blocks[block_id];
            if !block.must_be_in_block_eligible && self.skip_inelligible {
                continue;
            }
            let was_eligible = block.must_be_in_block_eligible;
            let mut row_required = vec![(1 << self.size) - 1; self.size];
            let mut col_required = vec![(1 << self.size) - 1; self.size];
            for joint_possibilities in &block.possibilities {
                let mut row_vals = vec![0; self.size];
                let mut col_vals = vec![0; self.size];
                for (i, x) in block.cells.iter().zip(joint_possibilities.iter()) {
                    row_vals[i / self.size] |= 1 << (x - 1);
                    col_vals[i % self.size] |= 1 << (x - 1);
                }
                for (m1, m2) in row_required.iter_mut().zip(row_vals.into_iter()) {
                    *m1 &= m2;
                }
                for (m1, m2) in col_required.iter_mut().zip(col_vals.into_iter()) {
                    *m1 &= m2;
                }
            }
            for i in 0..self.cells.len() {
                if self.cells[i].block_id == block_id {
                    continue;
                }
                let mask = !row_required[i / self.size] & !col_required[i % self.size];
                let cell_changed = self.mask_cell_possibilities(i, mask);
                made_progress |= cell_changed;
                assert!(
                    !cell_changed || was_eligible,
                    "Supposedly ineligible must be in block block made progress",
                );
            }
            self.blocks[block_id].must_be_in_block_eligible = false;
        }
        made_progress
    }
    // Finds subsets of values that must occur within a block. For example, if in a given row
    // there's one block that may contain a 3/6, and one block that must, and no other cells may
    // have a 3/6, the block that may contain a 3/6 must contain a 3/6.
    fn only_in_block(&mut self) -> bool {
        let mut made_progress = false;
        for transposed in [true, false] {
            for y in 0..self.size {
                // TODO: eligibility tracking
                made_progress |= self.only_in_block_single(transposed, y);
            }
        }
        made_progress
    }

    fn only_in_block_single(&mut self, transposed: bool, y: usize) -> bool {
        let size = self.size;
        let in_row: Box<dyn Fn(&usize) -> bool> = if transposed {
            Box::new(|ix| ix % size == y)
        } else {
            Box::new(|ix| ix / size == y)
        };
        let relevant_blocks = self
            .blocks
            .iter()
            .enumerate()
            .filter(|(_, block)| block.cells.iter().any(&in_row));
        // block_bitsets holds the possibilities of the currently relevant blocks, with cells
        // outside the current row ignored and flattened down to a set.
        let mut block_bitsets: Vec<_> = relevant_blocks
            .map(|(block_id, block)| {
                // TODO: some strange behaviour with colinear identical values in the same
                // possibiltiy, which is currently allowed. It probably won't affect correctness.
                let bitset_possibilities: Vec<_> = block
                    .possibilities
                    .iter()
                    .map(|possibility| {
                        zip(possibility, &block.cells)
                            .filter(|(_, ix)| in_row(ix))
                            .map(|(value, _)| (1 << value - 1))
                            .fold(0, Bitmask::bitor)
                    })
                    .collect();
                (block_id, bitset_possibilities)
            })
            .collect();
        let mut made_progress = false;
        for value_mask in 1..1 << self.size {
            // For every relevant block, match_counts stores the number of matches
            // acheivable across the different possibilities. For example, when matching
            // 2,4,6 against possibilities [(1,2),(2,4),(3,6)] the result will be {1,2}.
            // Matching 3,6 would yield {0,2}.
            // TODO: could we use bitsets instead? Yes, but this is already too
            // complicated. Make a proper bitset type before doing that.
            let match_counts: Vec<HashSet<u32>> = block_bitsets
                .iter()
                .map(|(_, bitsets)| {
                    bitsets
                        .iter()
                        .map(|bitset| (bitset & value_mask).count_ones())
                        .collect()
                })
                .collect();
            let forward_counts = running_possible_sums(match_counts.iter());
            let mut reverse_counts = running_possible_sums(match_counts.iter().rev());
            reverse_counts.reverse();
            for (i, (block_id, bitsets)) in block_bitsets.iter_mut().enumerate() {
                let prior_counts = &forward_counts[i];
                let following_counts = &reverse_counts[i + 1];
                let other_counts = possible_sums(prior_counts, following_counts);
                let mut to_remove = Vec::new();
                // Iterate backwards so indices are valid as we remove
                for (possibility_ix, bitset) in bitsets.iter().enumerate().rev() {
                    let possibility_count = (bitset & value_mask).count_ones();
                    if !other_counts.contains(&(value_mask.count_ones() - possibility_count)) {
                        to_remove.push(possibility_ix);
                    }
                }
                if to_remove.is_empty() {
                    continue;
                }
                let mut new_possibilities = self.blocks[*block_id].possibilities.clone();
                for j in to_remove {
                    new_possibilities.remove(j);
                    bitsets.remove(j);
                    // TODO: in principle, if we were accumulating prior_counts as we went,
                    // we could keep prior counts accurate, allowing us to remove more. Too
                    // complicated to start with though.
                }
                self.replace_block_possibilities(*block_id, new_possibilities);
                made_progress = true;
            }
        }
        made_progress
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
    fn most_interesting_block(&self) -> (usize, &BlockInfo) {
        self.blocks
            .iter()
            .enumerate()
            .filter(|(_, block)| block.radial_search_eligible && block.possibilities.len() > 1)
            .min_by_key(|(_, block)| {
                (
                    block.cells.len() * 4 * 3 * 5 / block.possibilities.len(),
                    block.cells.len(),
                )
            })
            .unwrap()
    }
    fn run_solver(&mut self, solver: Solver, stats: &mut Option<&mut SolversStats>) -> bool {
        let initial_entropy = self.entropy();
        let start = Instant::now();
        //dbg!(solver);
        let res = match solver {
            Solver::ExcludeNInN => self.exclude_n_in_n(),
            Solver::MustBeInBlock => self.must_be_in_block(),
            Solver::OnlyInBlock => self.only_in_block(),
            Solver::CompatibilitySearch => self.compatibility_search(),
            Solver::RadialSearchPromising => self.radial_search_promising(),
            Solver::RadialSearch => self.radial_search(),
        };
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

fn possible_sums(xs: &HashSet<u32>, ys: &HashSet<u32>) -> HashSet<u32> {
    let mut sums = HashSet::new();
    for x in xs {
        for y in ys {
            sums.insert(x + y);
        }
    }
    sums
}

fn running_possible_sums<'a>(it: impl Iterator<Item = &'a HashSet<u32>>) -> Vec<HashSet<u32>> {
    let zero_set: HashSet<u32> = [0].into_iter().collect();
    let mut out = vec![zero_set];
    for vals in it {
        let new_sums = possible_sums(vals, out.last().unwrap());
        out.push(new_sums);
    }
    out
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
        let table_data = solvers.into_iter().map(|solver| (solver, &self[solver]));

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
    fn test_only_in_block() {
        let save = std::fs::read("test_data/single_5_possibility").unwrap();
        let mut gs = GameState::from_save(save.as_slice());
        let block_id = gs.cells[5].block_id;
        dbg!(&gs.blocks[block_id]);
        assert_eq!(gs.blocks[block_id].possibilities.len(), 4);
        assert!(gs.only_in_block_single(true, 5));
        let mut possibilities = gs.blocks[block_id].possibilities.clone();
        possibilities.sort();
        assert_eq!(vec![vec![5, 6], vec![6, 5]], possibilities);
    }
}
