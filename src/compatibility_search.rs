use crate::game::GameState;

impl GameState<'_> {
    pub fn compatibility_search(&mut self) -> bool {
        let mut made_progress = false;
        for block_id in 0..self.blocks.len() {
            let eligible = self.blocks[block_id].compatibility_search_eligible;
            if !eligible && self.skip_inelligible {
                continue;
            }
            if self.compatibility_search_single(block_id) {
                made_progress = true;
                assert!(
                    eligible,
                    "Supposedly ineligible compatibility search block made progress",
                );
            }
        }
        made_progress
    }
    fn compatibility_search_single(&mut self, block_id: usize) -> bool {
        let block = &self.blocks[block_id];
        let old_joint_possibilities = &block.possibilities;
        // TODO: abort search when down to one possibility
        let new_joint_possibilities: Vec<_> = old_joint_possibilities
            .iter()
            .filter(|p| self.compatibility_search_inner(block_id, p))
            .cloned()
            .collect();
        let made_progress = self.replace_block_possibilities(block_id, new_joint_possibilities);
        self.blocks[block_id].compatibility_search_eligible = false;
        made_progress
    }

    // checks if every neighbor has a possibility compatiblle with the given possibility.
    fn compatibility_search_inner(
        &self,
        block_id: usize,
        block_joint_possibility: &[i8],
    ) -> bool {
        let block = &self.blocks[block_id];
        let res = block.interacting_blocks.iter().all(|&neighbor_id| {
            let neighbor = &self.blocks[neighbor_id];
            neighbor
                .possibilities
                .iter()
                .any(|neighbor_joint_possiblity| {
                    joint_possibilities_compatible(
                        self.size,
                        block_joint_possibility,
                        &block.cells,
                        neighbor_joint_possiblity,
                        &neighbor.cells,
                    )
                })
        });
        res
    }
}

fn joint_possibilities_compatible(
    size: usize,
    l_values: &[i8],
    l_cells: &[usize],
    r_values: &[i8],
    r_cells: &[usize],
) -> bool {
    for (x, i) in l_values.iter().zip(l_cells.iter()) {
        for (y, j) in r_values.iter().zip(r_cells.iter()) {
            if x == y && (i % size == j % size || i / size == j / size) {
                return false;
            }
        }
    }
    true
}
