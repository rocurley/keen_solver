use crate::game::Bitmask;

#[derive(Clone, Copy, Eq, PartialEq, Debug)]
// TODO: check this against Bitmask
pub struct BitMultiset(u32);

const MULTISET_WIDTH: u8 = 4;
const MULTISET_MASK: u32 = (1 << MULTISET_WIDTH) - 1;
impl BitMultiset {
    pub fn new() -> Self {
        BitMultiset(0)
    }
    pub fn set(&mut self, x: Bitmask) {
        self.0 |= 1 << (x * MULTISET_WIDTH);
    }
    #[cfg(test)]
    fn insert(&mut self, x: Bitmask) {
        self.0 += 1 << (x * MULTISET_WIDTH);
    }
    pub fn contains(&self, x: Bitmask) -> bool {
        let mask: u32 = MULTISET_MASK << (x * MULTISET_WIDTH);
        self.0 & mask > 0
    }
    #[cfg(test)]
    fn get(&self, x: Bitmask) -> u32 {
        (self.0 >> (x * MULTISET_WIDTH)) & MULTISET_MASK
    }
}

impl FromIterator<Bitmask> for BitMultiset {
    // Note that the semantics here are surprising: repeated instances of a number are ignored,
    // despite this being a multiset. This is because that's the behaviour nesecary to prevent
    // internal overflow in the solver.
    fn from_iter<T: IntoIterator<Item = Bitmask>>(iter: T) -> Self {
        let mut out = BitMultiset::new();
        for x in iter {
            out.set(x);
        }
        out
    }
}

#[cfg(test)]
pub fn possible_sums(xs: BitMultiset, ys: BitMultiset) -> BitMultiset {
    BitMultiset(xs.0 * ys.0)
}

pub fn possible_sums_iter(it: impl IntoIterator<Item = BitMultiset>) -> BitMultiset {
    BitMultiset(it.into_iter().map(|set| set.0).product())
}
pub fn undo_possible_sums(xs: BitMultiset, ys: BitMultiset) -> BitMultiset {
    BitMultiset(xs.0 / ys.0)
}

#[cfg(test)]
mod test {
    use std::collections::HashMap;

    use crate::game::Bitmask;

    use super::{possible_sums, BitMultiset};
    use prop::{collection::vec, num::u32};
    use proptest::prelude::*;
    fn new_multiset<'a, I: IntoIterator<Item = &'a Bitmask>>(iter: I) -> HashMap<Bitmask, u32> {
        let mut out = HashMap::new();
        for x in iter {
            *out.entry(*x).or_insert(0) = 1;
        }
        out
    }
    fn hash_possible_sums(
        xs: HashMap<Bitmask, u32>,
        ys: HashMap<Bitmask, u32>,
    ) -> HashMap<Bitmask, u32> {
        let mut out = HashMap::new();
        for (x, cx) in &xs {
            for (y, cy) in &ys {
                *out.entry(x + y).or_insert(0) += cx * cy;
            }
        }
        out
    }
    proptest! {
        #[test]
        fn test_get_contains(bits in u32::ANY, ix in 0u8..8) {
            let multiset = BitMultiset(bits);
            assert_eq!(multiset.get(ix) > 0, multiset.contains(ix));
        }
    }
    proptest! {
        #[test]
        fn test_possible_sums(elements1 in vec(0u8..4, 0..4), elements2 in vec(0u8..4, 0..4)) {
            let bit_multiset_1 : BitMultiset = elements1.iter().copied().collect();
            let bit_multiset_2 : BitMultiset = elements2.iter().copied().collect();
            let hash_multiset_1 = new_multiset(&elements1);
            let hash_multiset_2 = new_multiset(&elements2);
            let hash_merged = hash_possible_sums(hash_multiset_1, hash_multiset_2);
            let bit_merged = possible_sums(bit_multiset_1, bit_multiset_2);
            for (k, v) in hash_merged {
                assert_eq!(v, bit_merged.get(k), "key: {}, multiset: {:0b}", k, bit_merged.0);
            }
        }
    }
}
