use crate::{
    bitset::{possible_sums_iter, undo_possible_sums, BitMultiset},
    game::{Bitmask, GameState},
    load_with_default,
};
use std::{
    iter::zip,
    ops::BitOr,
    simd::{cmp::SimdPartialEq, num::SimdUint, LaneCount, Simd, SupportedLaneCount},
};

const LANES: usize = 8;
const SENTINEL: u8 = u8::MAX;
const SENTINEL_VEC: Simd<u8, LANES> = Simd::from_array([SENTINEL; LANES]);
struct BlockBitsets {
    block_id: usize,
    original_bitsets: Vec<u8>,
    compacted_bitsets: Simd<u8, LANES>,
}

fn simd_count_ones<const N: usize>(xs: &mut Simd<u32, N>)
where
    LaneCount<N>: SupportedLaneCount,
{
    for x in xs.as_mut_array() {
        *x = x.count_ones();
    }
}

impl GameState {
    fn only_in_block_eligilble(&mut self, y: usize, transposed: bool) -> &mut bool {
        if transposed {
            &mut self.cols_only_in_block_eligible[y]
        } else {
            &mut self.rows_only_in_block_eligible[y]
        }
    }
    // Finds subsets of values that must occur within a block. For example, if in a given row
    // there's one block that may contain a 3/6, and one block that must, and no other cells may
    // have a 3/6, the block that may contain a 3/6 must contain a 3/6.
    pub fn only_in_block(&mut self) -> bool {
        let skip_inelligible = self.skip_inelligible;
        for transposed in [true, false] {
            for y in 0..self.size {
                let eligibility = self.only_in_block_eligilble(y, transposed);
                if !*eligibility && skip_inelligible {
                    continue;
                }
                let was_eligible = *eligibility;
                *eligibility = false;
                if self.only_in_block_single(transposed, y) {
                    assert!(
                        was_eligible,
                        "Supposedly ineligible only_in_block row made progress",
                    );
                    return true;
                }
            }
        }
        false
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
        let mut block_bitsets = Vec::new();
        let mut match_sets_scratch = Vec::new();
        for (block_id, block) in relevant_blocks {
            // TODO: some strange behaviour with colinear identical values in the same
            // possibiltiy, which is currently allowed. It probably won't affect correctness.
            let bitset_possibilities: Vec<_> = block
                .possibilities
                .iter()
                .map(|possibility| {
                    zip(possibility, &block.cells)
                        .filter(|(_, ix)| in_row(ix))
                        .map(|(value, _)| (1 << (value - 1)))
                        .fold(0, Bitmask::bitor)
                })
                .collect();
            let mut compacted_bitsets = bitset_possibilities.clone();
            compacted_bitsets.sort_unstable();
            compacted_bitsets.dedup();
            if compacted_bitsets.len() > LANES {
                // 98th percentile
                return false;
            }
            if compacted_bitsets.contains(&SENTINEL) {
                return false;
            }
            block_bitsets.push(BlockBitsets {
                block_id,
                original_bitsets: bitset_possibilities,
                compacted_bitsets: load_with_default(&compacted_bitsets, SENTINEL),
            });
        }
        let mut made_progress = false;
        // A value mask has the same effect as its negation: counting even numbers is the same as
        // counting odd ones. This lets us stop halfway through the space.
        for value_mask in 1..=1 << (self.size - 1) {
            only_in_block_inner(&mut block_bitsets, value_mask, &mut match_sets_scratch);
        }
        for block in block_bitsets {
            let to_remove =
                block
                    .original_bitsets
                    .iter()
                    .enumerate()
                    .filter_map(|(i, bitset)| {
                        let bitset = Simd::splat(*bitset);
                        if block.compacted_bitsets.simd_eq(bitset).any() {
                            None
                        } else {
                            Some(i)
                        }
                    });
            made_progress |= self.delete_block_possibilities(block.block_id, to_remove);
        }
        made_progress
    }
}

fn only_in_block_inner(
    block_bitsets: &mut [BlockBitsets],
    value_mask: u8,
    match_sets: &mut Vec<BitMultiset>,
) {
    // For every relevant block, match_counts stores the number of matches
    // acheivable across the different possibilities. For example, when matching
    // 2,4,6 against possibilities [(1,2),(2,4),(3,6)] the result will be {1,2}.
    // Matching 3,6 would yield {0,2}.
    let value_mask_vec = Simd::splat(value_mask);
    match_sets.truncate(0);
    match_sets.extend(block_bitsets.iter().map(|block| {
        let mask = SENTINEL_VEC.simd_ne(block.compacted_bitsets);
        let mut match_counts: Simd<u32, LANES> =
            (block.compacted_bitsets & value_mask_vec).cast();
        simd_count_ones(&mut match_counts);
        BitMultiset::from_simd(match_counts, mask.cast())
    }));
    let target_count = Simd::splat(value_mask.count_ones());
    let total_counts = possible_sums_iter(match_sets.iter().copied());
    for (this_counts, block) in zip(&*match_sets, block_bitsets.iter_mut()) {
        let other_counts = undo_possible_sums(total_counts, *this_counts);
        let mut match_counts = (block.compacted_bitsets & value_mask_vec).cast();
        simd_count_ones(&mut match_counts);
        let needed_count = target_count - match_counts;
        let retain = other_counts.simd_contains(needed_count);
        block.compacted_bitsets = retain.cast().select(block.compacted_bitsets, SENTINEL_VEC);
    }
}

#[cfg(test)]
mod tests {
    use crate::GameState;
    #[test]
    fn test_only_in_block() {
        let save = std::fs::read("test_data/single_5_possibility").unwrap();
        let mut gs = GameState::from_save(save.as_slice());
        let block_id = gs.cells.block_id[5];
        dbg!(&gs.blocks[block_id]);
        assert_eq!(gs.blocks[block_id].possibilities.len(), 4);
        assert!(gs.only_in_block_single(true, 5));
        let mut possibilities = gs.blocks[block_id].possibilities.clone();
        possibilities.sort();
        assert_eq!(vec![vec![5, 6], vec![6, 5]], possibilities);
    }
}
