use std::{
    iter::Product,
    ops::{Div, DivAssign, Mul},
    simd::{
        cmp::{SimdPartialEq, SimdPartialOrd},
        num::SimdInt,
        Simd,
    },
};

// We only support up to size 9
const PRIMES: Simd<i32, 4> = Simd::from_array([2, 3, 5, 7]);

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Factorization(Simd<i8, 4>);

pub fn factorize(mut x: i32) -> Factorization {
    assert_ne!(x, 0);
    let mut out = Simd::splat(0);
    for (i, &p) in PRIMES.as_array().iter().enumerate() {
        while x % p == 0 {
            x /= p;
            out[i] += 1;
        }
    }
    Factorization(out)
}

const ZEROS: Simd<i8, 4> = Simd::from_array([0; 4]);
pub const ONE: Factorization = Factorization(ZEROS);

impl Factorization {
    // TODO: test this: seems easy to mess up.
    pub fn product(self) -> i32 {
        let ones = Simd::splat(1);
        let mut out = Simd::splat(1);
        let mut exp = self.0.cast::<i32>();
        let mut p = PRIMES;
        while exp.is_positive().any() {
            // p^exp = (p^2)^(exp / 2) * p ^ (exp & 1)
            let is_odd = (exp & ones).is_positive();
            out *= is_odd.select(p, ones);
            p *= p;
            exp >>= 1;
        }
        out.reduce_product()
    }
    pub fn is_whole(self) -> bool {
        !self.0.is_negative().any()
    }
    pub fn divides(self, other: Self) -> bool {
        self.0.simd_le(other.0).all()
    }
    // Rank 0: divisors of 7
    // Rank 1: divisors of 5 (and not 7)
    // etc
    pub fn rank(self) -> usize {
        self.0.simd_ne(ZEROS).to_bitmask().leading_zeros() as usize - 60
    }
}

impl Mul for Factorization {
    type Output = Self;
    #[allow(clippy::suspicious_arithmetic_impl)]
    fn mul(self, rhs: Self) -> Self {
        Factorization(self.0 + rhs.0)
    }
}

impl Product for Factorization {
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(ONE, Factorization::mul)
    }
}

impl Div for Factorization {
    type Output = Self;
    #[allow(clippy::suspicious_arithmetic_impl)]
    fn div(self, rhs: Self) -> Self {
        Factorization(self.0 - rhs.0)
    }
}

impl DivAssign for Factorization {
    fn div_assign(&mut self, rhs: Self) {
        *self = *self / rhs;
    }
}

#[cfg(test)]
mod tests {
    use super::{factorize, Factorization};
    use proptest::prelude::*;
    use std::array;
    use std::simd::Simd;

    proptest! {
        #[test]
        fn test_product_factorization(f_arr in array::from_fn(|_|0i8..5)) {
            let f = Factorization(Simd::from_array(f_arr));
            let p = f.product();
            let f2 = factorize(p);
            assert_eq!(f, f2, "product: {}", p);
        }
    }
}
