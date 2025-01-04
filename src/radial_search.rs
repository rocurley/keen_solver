use std::{
    ops::Range,
    simd::{cmp::SimdPartialEq, LaneCount, Mask, Simd, SupportedLaneCount},
};

use crate::game::{Bitmask, BlockInfo, GameState};

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
    fn current_possibility(&self) -> &[i8] {
        &self.block.possibilities[self.possibility_ix]
    }
}

struct SearchBlockSimd<'a, const LANES: usize>
where
    LaneCount<LANES>: SupportedLaneCount,
{
    block: &'a BlockInfo,
    possibility_ix: usize,
    interactions: Vec<<Simd<Bitmask, LANES> as SimdPartialEq>::Mask>,
    cell_range: Range<usize>,
}

impl<'a, const LANES: usize> SearchBlockSimd<'a, LANES>
where
    LaneCount<LANES>: SupportedLaneCount,
{
    fn can_increment(&self) -> bool {
        self.possibility_ix + 1 < self.block.possibilities.len()
    }
    fn current_possibility(&self) -> &[i8] {
        &self.block.possibilities[self.possibility_ix]
    }
    fn write_possibility(&self, cells: &mut Simd<i8, LANES>) {
        let target = &mut cells.as_mut_array()[self.cell_range.clone()];
        target.clone_from_slice(self.current_possibility());
    }
}

impl GameState {
    pub fn radial_search(&mut self) -> bool {
        let mut made_progress = false;
        for block_id in 0..self.blocks.len() {
            let eligible = self.blocks[block_id].radial_search_eligible;
            if !eligible && self.skip_inelligible {
                continue;
            }
            if self.radial_search_single(block_id, true) {
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
        self.radial_search_single(block_id, true)
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
    fn radial_search_single(&mut self, block_id: usize, vectorize: bool) -> bool {
        let mut new_possibilities = Vec::new();
        let possibilities = &self.blocks[block_id].possibilities;
        for (i, p) in possibilities.iter().enumerate() {
            // We assume that the puzzle is valid, so we can just keep the last possibility if
            // we've eliminated all others.
            if i == possibilities.len() - 1 && new_possibilities.is_empty() {
                new_possibilities.push(p.clone());
                continue;
            }
            if !self.radial_search_single_possibility(block_id, i, vectorize) {
                continue;
            }
            new_possibilities.push(p.clone());
        }
        let made_progress = self.replace_block_possibilities(block_id, new_possibilities);
        self.blocks[block_id].radial_search_eligible = false;
        made_progress
    }

    fn radial_search_single_possibility(
        &self,
        block_id: usize,
        possibility_ix: usize,
        vectorize: bool,
    ) -> bool {
        let block = &self.blocks[block_id];
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
        let cells_count: usize = search_space.iter().map(|sb| sb.block.cells.len()).sum();
        if vectorize {
            match cells_count {
                0..=8 => search_vectorized::<8>(search_space),
                9..=16 => search_vectorized::<16>(search_space),
                17..=32 => search_vectorized::<32>(search_space),
                _ => search_unvectorized(search_space),
            }
        } else {
            search_unvectorized(search_space)
        }
    }
}

fn search_unvectorized(mut search_space: Vec<SearchBlock<'_>>) -> bool {
    loop {
        let validation_failure = (1..search_space.len()).find(|&i| {
            let r_block = &search_space[i];
            r_block.interactions.iter().any(|interaction| {
                let local_val = r_block.current_possibility()[interaction.local_cell_ix];
                let other_val = search_space[interaction.other_block_ix].current_possibility()
                    [interaction.other_cell_ix];
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

fn search_vectorized<const LANES: usize>(old_search_space: Vec<SearchBlock<'_>>) -> bool
where
    LaneCount<LANES>: SupportedLaneCount,
{
    let mut search_space = Vec::<SearchBlockSimd<LANES>>::new();
    let mut start = 0;
    let mut cells = Simd::<i8, LANES>::splat(-1);
    for SearchBlock {
        block,
        possibility_ix,
        interactions,
    } in old_search_space
    {
        let mut simd_interactions = vec![Mask::splat(false); block.cells.len()];
        for interaction in interactions {
            let other_block = &search_space[interaction.other_block_ix];
            let i = other_block.cell_range.start + interaction.other_cell_ix;
            simd_interactions[interaction.local_cell_ix].set(i, true);
        }
        let end = start + block.cells.len();
        let cell_range = start..end;
        start = end;
        let sb = SearchBlockSimd {
            block,
            possibility_ix,
            interactions: simd_interactions,
            cell_range,
        };
        sb.write_possibility(&mut cells);
        search_space.push(sb);
    }
    loop {
        let validation_failure = (1..search_space.len()).find(|&i| {
            let r_block = &search_space[i];
            r_block
                .interactions
                .iter()
                .zip(r_block.cell_range.clone())
                .any(|(interaction, cell_i)| {
                    let cell_value = Simd::splat(cells[cell_i]);
                    let conflicts = *interaction & cell_value.simd_eq(cells);
                    conflicts.any()
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
        search_space[increment_point].write_possibility(&mut cells);
        for block in &mut search_space[increment_point + 1..] {
            block.possibility_ix = 0;
            block.write_possibility(&mut cells);
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::GameState;
    #[test]
    fn test_radial_search_vectorization() {
        let path = "test_data/radial_input";
        let save = std::fs::read(&path).unwrap();
        let mut expected = GameState::from_save(save.as_slice());
        let (block_id, _) = expected.most_interesting_block();
        let expected_progressed = expected.radial_search_single(block_id, false);
        let mut actual = expected.clone();
        let actual_progressed = actual.radial_search_single(block_id, true);
        assert_eq!(
            expected, actual,
            "{}\nexpected_progressed: {}\nactual_progressed: {}",
            path, expected_progressed, actual_progressed
        );
    }
}
