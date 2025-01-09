use std::{
    iter::zip,
    ops::BitOr,
    simd::{cmp::SimdPartialEq, LaneCount, Simd, SupportedLaneCount},
};

use crate::game::{BlockInfo, GameState};

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
    possibility_masks: Vec<Simd<u64, LANES>>,
}

impl<'a, const LANES: usize> SearchBlockSimd<'a, LANES>
where
    LaneCount<LANES>: SupportedLaneCount,
{
    fn can_increment(&self) -> bool {
        self.possibility_ix + 1 < self.block.possibilities.len()
    }
    fn current_possibility_mask(&self) -> Simd<u64, LANES> {
        self.possibility_masks[self.possibility_ix]
    }
    fn new(size: usize, block: &'a BlockInfo, possibility_ix: usize) -> Self {
        let possibility_masks = block
            .possibilities
            .iter()
            .map(|p| {
                let mut mask = Simd::<u64, LANES>::splat(0);
                for (&value, ix) in zip(p, &block.cells) {
                    let i = (ix % size) * size + value as usize - 1;
                    mask.as_mut_array()[i / 64] |= 1 << (i % 64);
                    let i = (size + ix / size) * size + value as usize - 1;
                    mask.as_mut_array()[i / 64] |= 1 << (i % 64);
                }
                mask
            })
            .collect();
        SearchBlockSimd {
            block,
            possibility_ix,
            possibility_masks,
        }
    }
}

impl<'arena> GameState<'arena> {
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
        if vectorize {
            match 2 * self.size * self.size {
                0..=64 => self.radial_search_vectorized::<1>(block_id, possibility_ix),
                65..=128 => self.radial_search_vectorized::<2>(block_id, possibility_ix),
                129..=256 => self.radial_search_vectorized::<4>(block_id, possibility_ix),
                _ => panic!("Size {} larger than expected", self.size),
            }
        } else {
            self.radial_search_unvectorized(block_id, possibility_ix)
        }
    }

    fn radial_search_unvectorized(&self, block_id: usize, possibility_ix: usize) -> bool {
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

    fn radial_search_vectorized<const LANES: usize>(
        &self,
        block_id: usize,
        possibility_ix: usize,
    ) -> bool
    where
        LaneCount<LANES>: SupportedLaneCount,
    {
        let block = &self.blocks[block_id];
        let mut search_space = vec![SearchBlockSimd::new(self.size, block, possibility_ix)];
        for &i in &block.interacting_blocks {
            let block = &self.blocks[i];
            let sb = SearchBlockSimd::new(self.size, block, 0);
            search_space.push(sb);
        }
        let mut start_point = 1;
        let mut seen = search_space[0].current_possibility_mask();
        loop {
            let validation_failure = (start_point..search_space.len()).find(|&i| {
                let r_block = &search_space[i];
                let r_mask = r_block.current_possibility_mask();
                if (r_mask & seen).simd_ne(Simd::splat(0)).any() {
                    return true;
                }
                seen |= r_mask;
                false
            });
            let mut increment_point = match validation_failure {
                None => {
                    return true;
                }
                Some(x) => x,
            };
            // There's a conflict between search_space[increment_point] and some block in
            // search_space[..increment_point].
            let seen_range_end = increment_point;
            debug_assert!(seen == reduce_possibility_masks(&search_space[..seen_range_end]));
            while increment_point > 0 && !search_space[increment_point].can_increment() {
                increment_point -= 1;
            }
            if increment_point == 0 {
                return false;
            }
            start_point = increment_point;
            // Roll seen back to the start point
            seen &= !reduce_possibility_masks(&search_space[start_point..seen_range_end]);
            search_space[increment_point].possibility_ix += 1;
            for block in &mut search_space[increment_point + 1..] {
                block.possibility_ix = 0;
            }
        }
    }
}

fn reduce_possibility_masks<const LANES: usize>(
    sbs: &[SearchBlockSimd<LANES>],
) -> Simd<u64, LANES>
where
    LaneCount<LANES>: SupportedLaneCount,
{
    sbs.iter()
        .map(SearchBlockSimd::current_possibility_mask)
        .fold(Simd::splat(0), Simd::bitor)
}

#[cfg(test)]
mod tests {
    use bumpalo::Bump;

    use crate::GameState;
    #[test]
    fn test_vectorization_small() {
        let path = "test_data/radial_input";
        single_vectorization_test(path);
    }
    #[ignore]
    #[test]
    fn test_vectorization_big() {
        for i in 0..=80 {
            let path = format!("hard_examples/{:04}a", i);
            single_vectorization_test(&path);
        }
    }

    fn single_vectorization_test(path: &str) {
        let save = std::fs::read(&path).unwrap();
        let arena = Bump::new();
        let mut expected = GameState::from_save(&arena, save.as_slice());
        let mut actual = expected.clone();
        let (block_id, _) = expected.most_interesting_block();
        let expected_progressed = expected.radial_search_single(block_id, false);
        let actual_progressed = actual.radial_search_single(block_id, true);
        assert_eq!(
            expected, actual,
            "{}\nexpected_progressed: {}\nactual_progressed: {}",
            path, expected_progressed, actual_progressed
        );
        /*
        let mut expected_bytes = Vec::new();
        expected.write_save(&mut expected_bytes);
        let mut actual_bytes = Vec::new();
        actual.write_save(&mut actual_bytes);
        assert_eq!(expected_bytes, actual_bytes);
        */
        assert_eq!(actual_progressed, expected_progressed);
    }
}
