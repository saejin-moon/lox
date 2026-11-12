use pyo3::prelude::*;

/// Fast action byte replay into NetHackHarness instance.
#[pyfunction]
pub fn fast_replay_action_bytes(
    py: Python,
    harness: PyObject,
    action_bytes: &[u8],
) -> PyResult<bool> {
    let nh = harness.getattr(py, "nh")?;
    let action_list = harness.getattr(py, "action_bytes")?;
    for &b in action_bytes {
        action_list.call_method1(py, "append", (b,))?;
        let res = nh.call_method1(py, "step", (b,))?;
        let done: bool = res.bind(py).get_item(1)?.extract::<bool>()?;
        if done {
            return Ok(true);
        }
    }
    Ok(false)
}
