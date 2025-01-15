#[cfg(test)]
pub fn visit_unique_permutations<T: Copy + Eq>(v: &[T], mut visit: impl FnMut(&[usize])) {
    let mut perm: Vec<_> = (0..v.len()).collect();
    let mut perm_inv: Vec<_> = (0..v.len()).collect();
    // perm[perm_inv[x]] == x
    loop {
        visit(&perm);
        // value to move in perm
        let mut x = 0;
        loop {
            let i = perm_inv[x];
            let j = i + 1;
            if j < v.len() && perm[i] < perm[j] && v[perm[i]] != v[perm[j]] {
                perm.swap(i, j);
                perm_inv.swap(perm[i], perm[j]);
                break;
            }
            // x has gone as far as it can go! We need to put it back where it started.
            // Since we only attempt to move x+1 after resetting x, we know all smaller values have
            // just been reset. This means we want to put in at the xth slot.
            if x == v.len() - 1 {
                return;
            }
            perm[x..=i].rotate_right(1);
            // TODO: Probably this is a left rotation or something
            for (i, &x) in perm.iter().enumerate() {
                perm_inv[x] = i;
            }
            x += 1;
        }
    }
}

#[cfg(test)]
pub fn unique_permutations<T: Copy + Eq>(v: &[T]) -> Vec<Vec<T>> {
    let mut out = Vec::new();
    let out_mut = &mut out;
    let visit = move |perm: &[usize]| {
        out_mut.push(permute(v, perm));
    };
    visit_unique_permutations(v, visit);
    out
}

#[cfg(test)]
pub fn visit_plain_changes<T: Copy>(v: &[T], mut visit: impl FnMut(&[usize])) {
    let mut perm: Vec<_> = (0..v.len()).collect();
    let mut perm_inv: Vec<_> = (0..v.len()).collect();
    // perm[perm_inv[i]] == i
    let mut direction = vec![true; v.len()];
    loop {
        visit(&perm);
        // value to move in perm
        let mut x = 0;
        loop {
            let i = perm_inv[x];
            let j = if direction[x] {
                Some(i + 1)
            } else {
                i.checked_sub(1)
            };
            if let Some(j) = j {
                if j < v.len() && perm[i] < perm[j] {
                    perm.swap(i, j);
                    perm_inv.swap(perm[i], perm[j]);
                    break;
                }
            }
            // x has gone as far as it can go!
            direction[x] = !direction[x];
            x += 1;
            if x == v.len() {
                return;
            }
        }
    }
}

#[cfg(test)]
fn plain_changes<T: Copy>(v: &[T]) -> Vec<Vec<T>> {
    let mut out = Vec::new();
    let out_mut = &mut out;
    let visit = move |perm: &[usize]| {
        out_mut.push(permute(v, perm));
    };
    visit_plain_changes(v, visit);
    out
}

pub fn visit_lexical_permutations<T: Ord>(v: &mut [T], mut visit: impl FnMut(&[T])) {
    v.sort_unstable();
    loop {
        visit(&*v);
        let Some((j, _)) = v
            .windows(2)
            .enumerate()
            .rev()
            .find(|(_, window)| window[0] < window[1])
        else {
            return;
        };
        let (l, _) = v
            .iter()
            .enumerate()
            .rev()
            .find(|(_, x)| **x > v[j])
            .unwrap();
        v.swap(j, l);
        v[j + 1..].reverse();
    }
}

#[cfg(test)]
fn lexical_permutations<T: Copy + Ord>(mut v: Vec<T>) -> Vec<Vec<T>> {
    let mut out = Vec::new();
    let out_mut = &mut out;
    let visit = move |v: &[T]| {
        out_mut.push(v.to_vec());
    };
    visit_lexical_permutations(&mut v, visit);
    out
}

#[cfg(test)]
pub fn permute<T: Copy>(v: &[T], perm: &[usize]) -> Vec<T> {
    let mut out = v.to_vec();
    for (i, &x) in perm.iter().enumerate() {
        out[i] = v[x];
    }
    out
}

#[cfg(test)]
fn recursive_permutations(v: &[i8]) -> Vec<Vec<i8>> {
    let mut out = Vec::new();
    let mut v: Vec<_> = v.to_vec();
    out.reserve((1..=v.len()).product());
    recursive_permutations_inner(&mut v, &mut out, 0);
    // Duplicates will be adjacent
    let has_duplicates = v.windows(2).any(|window| window[0] == window[1]);
    if has_duplicates {
        out.sort_unstable()
    }
    out.dedup();
    out
}

#[cfg(test)]
fn recursive_permutations_inner(v: &mut [i8], out: &mut Vec<Vec<i8>>, depth: usize) {
    if depth == v.len() {
        out.push(v.to_vec());
        return;
    }
    recursive_permutations_inner(v, out, depth + 1);
    for i in depth + 1..v.len() {
        v.swap(depth, i);
        recursive_permutations_inner(v, out, depth + 1);
        v.swap(depth, i);
    }
}

#[cfg(test)]
mod test {

    use crate::permutation::lexical_permutations;

    use super::{plain_changes, recursive_permutations, unique_permutations};

    #[test]
    fn test_plain_changes() {
        let v: Vec<_> = (0i8..5).collect();
        let mut actual = plain_changes(&v);
        for (i, permutation) in actual.iter().enumerate() {
            let mut sorted = permutation.clone();
            sorted.sort();
            assert_eq!(
                v, sorted,
                "{:?} (#{}) is not a permutation of {:?}",
                permutation, i, v
            );
        }
        let mut expected = recursive_permutations(&v);
        actual.sort();
        expected.sort();
        assert_eq!(expected, actual);
    }

    #[test]
    fn test_unique_permutations() {
        let v: Vec<_> = vec![0, 1, 1, 2, 3, 4, 4];
        let mut actual = unique_permutations(&v);
        for (i, permutation) in actual.iter().enumerate() {
            let mut sorted = permutation.clone();
            sorted.sort();
            assert_eq!(
                v, sorted,
                "{:?} (#{}) is not a permutation of {:?}",
                permutation, i, v
            );
        }
        let mut expected = recursive_permutations(&v);
        actual.sort();
        expected.sort();
        assert_eq!(expected, actual);
    }

    #[test]
    fn test_lexical_permutations() {
        let v: Vec<_> = vec![0, 1, 1, 2, 3, 4, 4];
        let mut actual = lexical_permutations(v.clone());
        for (i, permutation) in actual.iter().enumerate() {
            let mut sorted = permutation.clone();
            sorted.sort();
            assert_eq!(
                v, sorted,
                "{:?} (#{}) is not a permutation of {:?}",
                permutation, i, v
            );
        }
        let mut expected = recursive_permutations(&v);
        actual.sort();
        expected.sort();
        assert_eq!(expected, actual);
    }
}
