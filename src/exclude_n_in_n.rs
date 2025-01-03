use crate::game::{Bitmask, GameState};
use std::{
    cmp::Ordering,
    simd::{cmp::SimdPartialEq, num::SimdUint, Mask, Simd},
};

impl GameState {
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

    fn exclude_n_in_n_single(&mut self, transposed: bool, y: usize) -> bool {
        let mut row = self.get_row(y, transposed);
        for cell_mask in 1..1 << self.size {
            let cell_mask_vec = Mask::from_bitmask(cell_mask);
            let masked = cell_mask_vec.select(row.possibilities, Self::ZERO);
            let seen = masked.reduce_or();
            let unseen_vec = Simd::splat(!seen);
            match Bitmask::count_ones(seen).cmp(&cell_mask.count_ones()) {
                Ordering::Less => {
                    self.print_save();
                    dbg!(transposed, y);
                    eprintln!("mask: {:#06b}", cell_mask);
                    panic!("fewer possibilities than cells");
                }
                Ordering::Equal => {
                    row.possibilities =
                        cell_mask_vec.select(row.possibilities, row.possibilities & unseen_vec);
                }
                Ordering::Greater => {}
            }
        }
        self.update_row(row)
    }

    const ZERO: Simd<Bitmask, 8> = Simd::<Bitmask, 8>::from_array([0; 8]);

    fn exclude_n_in_n_single_dual(&mut self, transposed: bool, y: usize) -> bool {
        let mut row = self.get_row(y, transposed);
        for value_mask in 1..1 << self.size {
            let vec_mask = Simd::<Bitmask, 8>::splat(value_mask);
            let matches = vec_mask & row.possibilities;
            let match_mask = matches.simd_ne(Self::ZERO);
            match match_mask
                .to_bitmask()
                .count_ones()
                .cmp(&value_mask.count_ones())
            {
                Ordering::Less => {
                    self.print_save();
                    dbg!(transposed, y);
                    eprintln!("mask: {:#06b}", value_mask);
                    panic!("fewer cells than values");
                }
                Ordering::Equal => {
                    row.possibilities = match_mask.select(matches, row.possibilities);
                }
                Ordering::Greater => {}
            }
        }
        self.update_row(row)
    }
}
