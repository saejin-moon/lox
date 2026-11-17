use pyo3::prelude::*;

mod blackboard;
mod bt;
mod path;
mod replay;

pub use blackboard::PyBlackboard;
pub use bt::{
    PyActionLeaf, PyBehaviorTree, PyConditionLeaf, PyNodeStatus, PyPrioritySelector, PySequence,
};
pub use path::{fast_astar_next_step, fast_bfs_distance_grid};
pub use replay::{
    fast_extract_exits, fast_replay_action_bytes, fast_trap_mask,
    fast_unexplored_frontier_mask, fast_update_floor_memory,
};

/// High-throughput LOX Native Core module (`lox_core`).
#[pymodule]
fn lox_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyNodeStatus>()?;
    m.add_class::<PyBlackboard>()?;
    m.add_class::<PyActionLeaf>()?;
    m.add_class::<PyConditionLeaf>()?;
    m.add_class::<PyPrioritySelector>()?;
    m.add_class::<PySequence>()?;
    m.add_class::<PyBehaviorTree>()?;

    m.add_function(wrap_pyfunction!(fast_bfs_distance_grid, m)?)?;
    m.add_function(wrap_pyfunction!(fast_astar_next_step, m)?)?;
    m.add_function(wrap_pyfunction!(fast_replay_action_bytes, m)?)?;
    m.add_function(wrap_pyfunction!(fast_update_floor_memory, m)?)?;
    m.add_function(wrap_pyfunction!(fast_extract_exits, m)?)?;
    m.add_function(wrap_pyfunction!(fast_trap_mask, m)?)?;
    m.add_function(wrap_pyfunction!(fast_unexplored_frontier_mask, m)?)?;

    Ok(())
}
