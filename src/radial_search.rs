use crate::game::{BlockInfo, GameState};

impl GameState {
    pub fn radial_search(&mut self) -> bool {
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

    pub fn radial_search_promising(&mut self) -> bool {
        let (block_id, _) = self.most_interesting_block();
        self.radial_search_single(block_id)
    }

    fn most_interesting_block(&self) -> (usize, &BlockInfo) {
        self.blocks
            .iter()
            .enumerate()
            .filter(|(_, block)| block.radial_search_eligible && block.possibilities.len() > 1)
            .min_by_key(|(_, block)| {
                (
                    // Why the multiplication by a constant? I think it's to let us use integer
                    // division without rounding error.
                    block.cells.len() * 4 * 3 * 5 / block.possibilities.len(),
                    block.cells.len(),
                )
            })
            .unwrap()
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
}
