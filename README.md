# Keen Solver

Solves keen puzzles of size 8 and below, as fast as possible. This is useless, since there's no need to solve keen puzzles in 60 microseconds (and it doesn't show you the solution), but it's a fun challenge.

Accepts one game ID (in the format used by [Simon Tatham's puzzle collection](https://www.chiark.greenend.org.uk/~sgtatham/puzzles/js/keen.html)) per line in stdin.
Doesn't actually print anything per puzzle unless it fails to solve the puzzle, in which case it will print a save file showing how far it got on the puzzle it failed to solve.
`--stats` will print some performance statistics. `--trace` will print a trace of which solvers were used.

## How does it work?

After parsing a game ID and setting up the internal representation of an unsolved puzzle, `keen_solver` iterates through 7 solvers, each of which attempt to make progress.
If one of them makes progress, it goes back to the beginning.
These solves are arranged from most to least efficient: `keen_solver` tries cheap solvers before falling back to more expensive ones as needed.

Here's a sample run over 100,000 puzzles, showing the performance of the different solvers:

| solver                  |     calls | successes | entropy_removed | duration     | throughput (kbps) |
| ----------------------- | --------: | --------: | --------------: | ------------ | ----------------: |
| `ExcludeNInN`           | `1169755` |  `773696` |     `3575947.8` | 2.186303943s |            `1636` |
| `MustBeInBlock`         |  `396059` |   `75127` |       `92070.2` | 264.713098ms |             `348` |
| `OnlyInBlock`           |  `320932` |  `303922` |      `623891.4` | 2.260324323s |             `276` |
| `CompatibilitySearch`   |   `17010` |    `9315` |       `16387.2` | 345.746375ms |              `47` |
| `RadialSearchPromising` |    `7695` |    `7677` |       `24430.8` | 422.279336ms |              `58` |
| `RadialSearch`          |      `18` |      `18` |         `258.6` | 563.981µs    |             `459` |

The different solvers are described below.

## `ExcludeNInN`

This solver finds cases where n cells in a row (or col) share the same n possibilities.
For example, two cells in a row that can both be either 2 or 3.
Those n possibilities must be found in those n cells, so this solver excludes them from other cells in the same row (col).
As a special case, this also handles the case where a cell is known with certainty, removing the that value as a possibility from other cells in the same row and column.

This is where the vast majority of the possibility space is eliminated.

This solver is particularly amenable to vectorization.
A vector of bitsets representing possible values for every cell is loaded into a 512 bit SIMD register, allowing the entire board state for this solver to be stored in a single register, and for every row to be processed at the same time.

## `MustBeInBlock`

If a given block requires that some number be in that block in a specific row or column, this solver filters out that number from the rest of the row or column.
For example, a 2-cell block with the clue `/2` in a 6x6 puzzle has 3 possibilities (up to ordering): `[1,2],[2,4],[3,6]`.
If `[3,6]` has been otherwise eliminated, we know that there must be a 2 somewhere in the block.

This solver could probably be eliminated: it rarely succeeds, and isn't much faster than the next solver.
But since it's already written and it pulls its weight, it's still here.

## `OnlyInBlock`

For every row (col), this solver counts how many times a subset of values can occur in every block, and eliminates possibilities inconsistent with other blocks.
For example, a frequently useful subset is "even numbers".
In a 6x6 board, each row (col) must have 3 even numbers: 2,4, and 6.
If a row has 3 blocks of 2 cells each, with clues `-3`, `+8`, and `*12`, the possibilities are as follows:

| Block | Possibilities       | Number of even numbers |
| ----- | ------------------- | ---------------------- |
| `-3`  | `[1,4],[2,5],[3,6]` | 1                      |
| `+8`  | `[3,5], [2,6]`      | 0 or 2                 |
| `*12` | `[2,6], [3,4]`      | 1 or 2                 |

Since there must be 3 even numbers in total, it's impossible for `+8` to have 2 even numbers: that would make the minimum number of even numbers 1 + 2 + 1 = 4.
So this solver can determine that the `+8` block has a 3 and a 5.
This further implies that the `*12` block must have a 2 and a 6, since it must have 2 even numbers (although `ExcludeNInN` would also be able to determine this, by eliminating `3` and `5` from the`*12` block).

This solver is the second most productive one, and indeed, when I tested 10 puzzles this and `ExcludeNInN` were sufficient for 9 of them.
It's also the newest solver: its introduction has thrown the viability of a lot other solvers into question.

## `CompatibilitySearch`

This solver filters a block's possibilities by whether their interacting blocks have any compatible possibilities.
For example, consider the same example used for `OnlyInBlock`: a row with 3 blocks of 2 cells each, with clues `-3`, `+8`, and `*12`.
The "compatibility table" between `-3` and `+8` looks like this:

|         | `[3,5]` | `[2,6]` |
| ------- | ------- | ------- |
| `[1,4]` | ✅      | ✅      |
| `[2,5]` | ✅      | ❌      |
| `[3,6]` | ❌      | ❌      |

(In this case, where both blocks share a row, the possibilities are incompatible if they share any numbers).
This table makes it clear that `[3,6]` is not compatible with any possible values for `+8`, and therefore would be eliminated by `CompatibilitySearch`.

This example, like most examples for compatibility search, can also be solved by `OnlyInBlock`.
As such, compatibility search is rarely called, and has lukewarm success rates.
Furthermore, it's superficially slower than `RadialSearch` and `RadialSearchPromising`.
Before `OnlyInBlock` was introduced, this solver was significantly faster and more productive, but it's now starved for cases it can work well on by `OnlyInBlock`.
I plan to investigate removing this solver.

## `RadialSearchPromising`

This solver performs a "radial search" on the block that seems most promising, according to a simple heuristic.
"Radial search" is not a meaningful name: it's based on a flawed understanding I originally had of how this search works: don't read too much into it.
A radial search performs a backtracking search for any consistent assignment of values to the entire "neighborhood" of a block: all the blocks that share a row or column with the block in question.
It eliminates possibilities that have no such consistent assignment.

This solver is expensive, but almost always (~99.8% of the time) makes progress.
This may be unsurprising given that the neighborhood of a block can easily be half the board: this is one step short of a brute force scan of all remaining possible board states.

## `RadialSearch`

This solver performs a "radial search" (as above) on every block.
I have found no cases where it does not make progress, and so it's the solver of last resort.

Interestingly, after the addition of `OnlyInBlock` it shows up as faster than `RadialSearchPromising`.
I need to investigate that more: possibly `RadialSearchPromising` should be removed, or the heuristic tweaked.
