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

pub fn unique_permutations<T: Copy + Eq>(v: &[T]) -> Vec<Vec<T>> {
    let mut out = Vec::new();
    let out_mut = &mut out;
    let visit = move |perm: &[usize]| {
        out_mut.push(permute(v, perm));
    };
    visit_unique_permutations(v, visit);
    out
}

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

fn plain_changes<T: Copy>(v: &[T]) -> Vec<Vec<T>> {
    let mut out = Vec::new();
    let out_mut = &mut out;
    let visit = move |perm: &[usize]| {
        out_mut.push(permute(v, perm));
    };
    visit_plain_changes(v, visit);
    out
}

pub fn permute<T: Copy>(v: &[T], perm: &[usize]) -> Vec<T> {
    let mut out = v.to_vec();
    for (i, &x) in perm.iter().enumerate() {
        out[i] = v[x];
    }
    out
}

fn permutations(v: &[i8]) -> Vec<Vec<i8>> {
    let mut out = Vec::new();
    let mut v: Vec<_> = v.to_vec();
    out.reserve((1..=v.len()).product());
    permutations_inner(&mut v, &mut out, 0);
    // Duplicates will be adjacent
    let has_duplicates = v.windows(2).any(|window| window[0] == window[1]);
    if has_duplicates {
        out.sort_unstable()
    }
    out.dedup();
    out
}

fn permutations_inner<'arena>(v: &mut [i8], out: &mut Vec<Vec<i8>>, depth: usize) {
    if depth == v.len() {
        out.push(v.to_vec());
        return;
    }
    permutations_inner(v, out, depth + 1);
    for i in depth + 1..v.len() {
        v.swap(depth, i);
        permutations_inner(v, out, depth + 1);
        v.swap(depth, i);
    }
}

#[cfg(test)]
mod test {

    use super::{permutations, plain_changes, unique_permutations};

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
        let mut expected = permutations(&v);
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
        let mut expected = permutations(&v);
        actual.sort();
        expected.sort();
        assert_eq!(expected, actual);
    }
}
