use std::mem::size_of;
use std::ops::Shl;

use crate::game::Bitmask;

// Values start at 0
#[derive(Clone, Copy, Eq, PartialEq)]
pub struct Bitset0(Bitmask);

impl Bitset0 {
    pub fn new() -> Self {
        Bitset0(0)
    }
    pub fn insert<I>(&mut self, x: I)
    where
        Bitmask: Shl<I>,
        Bitmask: From<<Bitmask as Shl<I>>::Output>,
    {
        self.0 |= Bitmask::from(1 << x);
    }
    pub fn contains<I>(&self, x: I) -> bool
    where
        Bitmask: Shl<I>,
        Bitmask: From<<Bitmask as Shl<I>>::Output>,
    {
        self.0 & Bitmask::from(1 << x) > 0
    }
    fn iter<I>(self) -> impl Iterator<Item = I>
    where
        Bitmask: Shl<I>,
        Bitmask: From<<Bitmask as Shl<I>>::Output>,
        I: From<usize> + Copy,
    {
        (0..8 * size_of::<Bitmask>())
            .into_iter()
            .map(I::from)
            .filter(move |i| self.0 & Bitmask::from(1 << *i) > 0)
    }
}

impl<I> FromIterator<I> for Bitset0
where
    Bitmask: Shl<I>,
    Bitmask: From<<Bitmask as Shl<I>>::Output>,
{
    fn from_iter<T: IntoIterator<Item = I>>(iter: T) -> Self {
        let mut out = Bitset0::new();
        for x in iter {
            out.insert(x);
        }
        out
    }
}

// TODO: this is almost multiplication. Surely there's some clever way to do this.
pub fn possible_sums(xs: Bitset0, ys: Bitset0) -> Bitset0 {
    let mut sums = Bitset0::new();
    for x in xs.iter::<usize>() {
        sums.0 |= ys.0 << x;
    }
    sums
}

/*
// Significantly slower.
fn possible_sums_bad(xs: Bitset0, ys: Bitset0) -> Bitset0 {
    let mut sums = Bitset0::new();
    for i in 0..8 * size_of::<Bitmask>() {
        sums.0 |= ys.0 * (1 << i & xs.0)
    }
    sums
}
*/
