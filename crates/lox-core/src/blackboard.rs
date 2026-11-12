use pyo3::prelude::*;
use std::collections::HashMap;

/// High-throughput typed Blackboard for inter-node Behavior Tree state.
#[pyclass(name = "Blackboard", subclass)]
#[derive(Default)]
pub struct PyBlackboard {
    pub(crate) storage: HashMap<String, PyObject>,
}

#[pymethods]
impl PyBlackboard {
    #[new]
    pub fn new() -> Self {
        Self {
            storage: HashMap::with_capacity(32),
        }
    }

    #[pyo3(signature = (key, default = None))]
    pub fn get_var(&self, py: Python, key: &str, default: Option<PyObject>) -> PyObject {
        match self.storage.get(key) {
            Some(obj) => obj.clone_ref(py),
            None => default.unwrap_or_else(|| py.None()),
        }
    }

    pub fn set_var(&mut self, key: String, value: PyObject) {
        self.storage.insert(key, value);
    }

    pub fn has_var(&self, key: &str) -> bool {
        self.storage.contains_key(key)
    }

    pub fn clear_vars(&mut self) {
        self.storage.clear();
    }

    pub fn __getitem__(&self, py: Python, key: &str) -> PyResult<PyObject> {
        self.storage
            .get(key)
            .map(|obj| obj.clone_ref(py))
            .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err(key.to_string()))
    }

    pub fn __setitem__(&mut self, key: String, value: PyObject) {
        self.storage.insert(key, value);
    }

    pub fn __contains__(&self, key: &str) -> bool {
        self.storage.contains_key(key)
    }

    pub fn __len__(&self) -> usize {
        self.storage.len()
    }

    #[pyo3(signature = (key, default = None))]
    pub fn get(&self, py: Python, key: &str, default: Option<PyObject>) -> PyObject {
        self.get_var(py, key, default)
    }

    pub fn clear(&mut self) {
        self.clear_vars();
    }
}
