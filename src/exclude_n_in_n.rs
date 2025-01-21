use crate::game::{Bitmask, GameState};
use std::{
    cmp::Ordering,
    ops::BitOr,
    simd::{
        cmp::SimdPartialEq,
        num::{SimdInt, SimdUint},
        LaneCount, Mask, Simd, SupportedLaneCount,
    },
};

const LANES: usize = 8;
const ZERO: Simd<Bitmask, { LANES }> = Simd::from_array([0; LANES]);

impl GameState<'_> {
    fn exclude_n_in_n_eligible(&mut self, y: usize, transposed: bool) -> &mut bool {
        if transposed {
            &mut self.cols_exclude_n_in_n_eligible[y]
        } else {
            &mut self.rows_exclude_n_in_n_eligible[y]
        }
    }
    // It seems like there's a dual thing here. Currently, we:
    // * Pick out subsets of cells
    // * If there are n possibilities in n cells, those possibilities can be eliminated elsewhere.
    // Instead, we could:
    // * Pick out a subset of values
    // * If there are n values that only appear in n cells, then those cells can eliminate all
    // other possibilties
    // But these might be the same. If there are n cells with n possibilities, then the (s-n) other
    // cells will be the only place where the (s-n) other values occur.
    pub fn exclude_n_in_n(&mut self) -> bool {
        if self.size <= 8 {
            self.exclude_n_in_n_whole_board()
        } else {
            self.exclude_n_in_n_by_rows()
        }
    }
    pub fn exclude_n_in_n_by_rows(&mut self) -> bool {
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
                if self.exclude_n_in_n_single_dual(transposed, y) {
                    made_progress = true;
                    if !was_eligible {
                        self.print_save();
                        dbg!(transposed, y);
                        panic!("Supposedly ineligbile row/col made progress.");
                    }
                }
            }
        }
        made_progress
    }

    fn exclude_n_in_n_single_dual(&mut self, transposed: bool, y: usize) -> bool {
        let mut row = self.get_row(y, transposed);
        for value_mask in 1..1 << self.size {
            let vec_mask = Simd::splat(value_mask);
            let matches = vec_mask & row.possibilities;
            let match_mask = matches.simd_ne(ZERO);
            match match_mask
                .to_bitmask()
                .count_ones()
                .cmp(&value_mask.count_ones())
            {
                #[cfg(debug_assertions)]
                Ordering::Less => {
                    self.print_save();
                    dbg!(transposed, y);
                    eprintln!("mask: {:#06b}", value_mask);
                    panic!("fewer cells than values");
                }
                Ordering::Equal => {
                    row.possibilities = match_mask.select(matches, row.possibilities);
                }
                _ => {}
            }
        }
        self.update_row(row)
    }

    pub fn assert_board_vec_nonzero(&self, board: Simd<u8, 64>) {
        debug_assert!(board.as_array()[..self.size * self.size]
            .iter()
            .all(|x| *x > 0))
    }

    pub fn exclude_n_in_n_whole_board(&mut self) -> bool {
        assert!(self.size <= 8);
        let mut board = self.board_as_vec();
        self.assert_board_vec_nonzero(board);
        let zero = Simd::splat(0);
        let (row_masks, col_masks) = row_col_masks_reference(self.size);
        for transposed in [true, false] {
            let row_masks = if transposed { col_masks } else { row_masks };
            for value_mask in 1..1 << self.size {
                let vec_mask = Simd::splat(value_mask);
                let matches = vec_mask & board;
                let match_mask = matches.simd_ne(zero).to_bitmask();
                let row_match_counts = simd_count_ones(Simd::splat(match_mask) & row_masks);
                let triggered_rows =
                    row_match_counts.simd_eq(Simd::splat(value_mask.count_ones() as u64));
                let triggered_cells =
                    (triggered_rows.to_int().cast() & row_masks).reduce_or() & match_mask;
                let triggered_cells_vec = Mask::<i8, 64>::from_bitmask(triggered_cells);
                let new_board = triggered_cells_vec.select(matches, board);
                // Make this conditional to remove the data dependency in the most likely case
                if new_board != board {
                    board = new_board;
                }
                self.assert_board_vec_nonzero(board);
            }
        }
        self.update_board(board)
    }
}

fn row_col_masks_reference(size: usize) -> (Simd<u64, 8>, Simd<u64, 8>) {
    let ids: Vec<_> = (0..(size * size) as u8).collect();
    let ids: Simd<u8, 64> = Simd::load_or_default(&ids);
    let mut mask = [false; 64];
    mask[..size * size].copy_from_slice(&vec![true; size * size]);
    let mask = Mask::from_array(mask);
    let row = ids / Simd::splat(size as u8);
    let col = ids % Simd::splat(size as u8);
    let row_masks: Vec<_> = (0..size as u8)
        .map(|i| (mask & row.simd_eq(Simd::splat(i))).to_bitmask())
        .collect();
    let row_masks = Simd::load_or_default(&row_masks);
    let col_masks: Vec<_> = (0..size as u8)
        .map(|i| (mask & col.simd_eq(Simd::splat(i))).to_bitmask())
        .collect();
    let col_masks = Simd::load_or_default(&col_masks);
    (row_masks, col_masks)
}

fn simd_count_ones<const N: usize>(mut xs: Simd<u64, N>) -> Simd<u64, N>
where
    LaneCount<N>: SupportedLaneCount,
{
    for x in xs.as_mut_array() {
        *x = x.count_ones() as u64;
    }
    xs
}

#[cfg(test)]
mod tests {
    use std::{
        fs::File,
        io::{BufRead, BufReader},
    };

    use bumpalo::Bump;

    use crate::parse_game_id;

    #[test]
    fn test_exclude_n_in_n_whole_board() {
        use pretty_assertions::assert_eq;
        let n = 1;
        let file = File::open("puzzles").unwrap();
        let puzzles = BufReader::new(file).lines().take(n);
        for game_seed in puzzles {
            let game_seed = game_seed.unwrap();
            let arena = Bump::new();
            let game = parse_game_id(&arena, &game_seed);
            let mut expected = game.clone();
            expected.exclude_n_in_n_by_rows();
            let mut actual = game.clone();
            actual.exclude_n_in_n_whole_board();
            assert_ne!(game.cells.possibilities, actual.cells.possibilities);
            // Wrong condition. But whole_board should be slightly weaker than by_rows, because
            // by_rows bounces back to blocks every row, and whole_board does not.
            assert_eq!(expected.cells.possibilities, actual.cells.possibilities);
            assert_eq!(expected.blocks, actual.blocks);
        }
    }
}
