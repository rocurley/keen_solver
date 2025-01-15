use crate::game::GameState;
use std::simd::Simd;

impl GameState<'_> {
    // If a given block requires that some number be in that block in a specific row or column,
    // filter out that number from the rest of the row or column.
    // Example: filter out 2 from a row containing a /2 block with possibilities (1,2,4)
    pub fn must_be_in_block(&mut self) -> bool {
        const ROW_LANES: usize = 8;
        let mut made_progress = false;
        for block_id in 0..self.blocks.len() {
            let block = &self.blocks[block_id];
            if !block.must_be_in_block_eligible && self.skip_inelligible {
                continue;
            }
            let was_eligible = block.must_be_in_block_eligible;
            let mut row_required: Simd<u8, ROW_LANES> = Simd::splat((1 << self.size) - 1);
            let mut col_required: Simd<u8, ROW_LANES> = Simd::splat((1 << self.size) - 1);
            for joint_possibilities in &block.possibilities {
                let mut row_vals: Simd<u8, ROW_LANES> = Simd::splat(0);
                let mut col_vals: Simd<u8, ROW_LANES> = Simd::splat(0);
                for (i, x) in block.cells.iter().zip(joint_possibilities.iter()) {
                    row_vals[i / self.size] |= 1 << (x - 1);
                    col_vals[i % self.size] |= 1 << (x - 1);
                }
                row_required &= row_vals;
                col_required &= col_vals;
            }
            for y in 0..self.size {
                let row_mask = Simd::splat(!row_required[y]) & !col_required;
                for x in 0..self.size {
                    let i = self.size * y + x;
                    if self.cells.block_id[i] == block_id {
                        continue;
                    }
                    let mask = row_mask[x];
                    let cell_changed = self.mask_cell_possibilities(i, mask);
                    made_progress |= cell_changed;
                    assert!(
                        !cell_changed || was_eligible,
                        "Supposedly ineligible must be in block block made progress",
                    );
                }
            }
            self.blocks[block_id].must_be_in_block_eligible = false;
        }
        made_progress
    }
}
