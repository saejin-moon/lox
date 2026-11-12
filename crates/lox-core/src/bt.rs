use pyo3::prelude::*;
use crate::blackboard::PyBlackboard;

/// Node execution status for Behavior Tree nodes.
#[pyclass(name = "NodeStatus", eq, eq_int)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PyNodeStatus {
    SUCCESS,
    FAILURE,
    RUNNING,
}

#[pymethods]
impl PyNodeStatus {
    #[getter]
    fn value(&self) -> &'static str {
        match self {
            Self::SUCCESS => "SUCCESS",
            Self::FAILURE => "FAILURE",
            Self::RUNNING => "RUNNING",
        }
    }

    fn __repr__(&self) -> String {
        format!("NodeStatus.{}", self.value())
    }
}

/// Action leaf node that invokes an action function.
#[pyclass(name = "ActionLeaf", subclass)]
pub struct PyActionLeaf {
    pub(crate) action_fn: PyObject,
    pub(crate) name: String,
    pub(crate) is_running: bool,
}

#[pymethods]
impl PyActionLeaf {
    #[new]
    #[pyo3(signature = (action_fn, name = "".to_string(), is_running = false))]
    pub fn new(action_fn: PyObject, name: String, is_running: bool) -> Self {
        Self {
            action_fn,
            name,
            is_running,
        }
    }

    #[getter]
    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn tick(
        &self,
        py: Python,
        state: PyObject,
        blackboard: PyObject,
    ) -> PyResult<(PyNodeStatus, Option<PyObject>)> {
        let res = match self.action_fn.call1(py, (&state, &blackboard)) {
            Ok(val) => val,
            Err(e) => {
                if e.is_instance_of::<pyo3::exceptions::PyTypeError>(py) {
                    match self.action_fn.call1(py, (&state,)) {
                        Ok(val) => val,
                        Err(inner_err) => return Err(inner_err),
                    }
                } else {
                    return Err(e);
                }
            }
        };

        if res.is_none(py) {
            Ok((PyNodeStatus::FAILURE, None))
        } else {
            let status = if self.is_running {
                PyNodeStatus::RUNNING
            } else {
                PyNodeStatus::SUCCESS
            };
            Ok((status, Some(res)))
        }
    }
}

/// Condition leaf node that evaluates a boolean precondition.
#[pyclass(name = "ConditionLeaf", subclass)]
pub struct PyConditionLeaf {
    pub(crate) cond_fn: PyObject,
    pub(crate) name: String,
}

#[pymethods]
impl PyConditionLeaf {
    #[new]
    #[pyo3(signature = (cond_fn, name = "".to_string()))]
    pub fn new(cond_fn: PyObject, name: String) -> Self {
        Self { cond_fn, name }
    }

    #[getter]
    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn tick(
        &self,
        py: Python,
        state: PyObject,
        blackboard: PyObject,
    ) -> PyResult<(PyNodeStatus, Option<PyObject>)> {
        let res = match self.cond_fn.call1(py, (&state, &blackboard)) {
            Ok(val) => val,
            Err(e) => {
                if e.is_instance_of::<pyo3::exceptions::PyTypeError>(py) {
                    match self.cond_fn.call1(py, (&state,)) {
                        Ok(val) => val,
                        Err(inner_err) => return Err(inner_err),
                    }
                } else {
                    return Err(e);
                }
            }
        };

        let is_ok: bool = res.extract(py).unwrap_or(false);
        let status = if is_ok {
            PyNodeStatus::SUCCESS
        } else {
            PyNodeStatus::FAILURE
        };
        Ok((status, None))
    }
}

/// Priority Selector (Fallback): ticks children sequentially until one does not fail.
#[pyclass(name = "PrioritySelector", subclass)]
pub struct PyPrioritySelector {
    pub(crate) children: Vec<PyObject>,
    pub(crate) name: String,
}

#[pymethods]
impl PyPrioritySelector {
    #[new]
    #[pyo3(signature = (children = None, name = "".to_string()))]
    pub fn new(children: Option<Vec<PyObject>>, name: String) -> Self {
        Self {
            children: children.unwrap_or_default(),
            name,
        }
    }

    #[getter]
    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn add_child(&mut self, child: PyObject) {
        self.children.push(child);
    }

    pub fn tick(
        &self,
        py: Python,
        state: PyObject,
        blackboard: PyObject,
    ) -> PyResult<(PyNodeStatus, Option<PyObject>)> {
        for child in &self.children {
            let res = child.call_method1(py, "tick", (&state, &blackboard))?;
            let (status, act): (PyNodeStatus, Option<PyObject>) = res.extract(py)?;
            if status == PyNodeStatus::SUCCESS || status == PyNodeStatus::RUNNING {
                return Ok((status, act));
            }
        }
        Ok((PyNodeStatus::FAILURE, None))
    }
}

/// Sequence: ticks children sequentially until one fails.
#[pyclass(name = "Sequence", subclass)]
pub struct PySequence {
    pub(crate) children: Vec<PyObject>,
    pub(crate) name: String,
}

#[pymethods]
impl PySequence {
    #[new]
    #[pyo3(signature = (children = None, name = "".to_string()))]
    pub fn new(children: Option<Vec<PyObject>>, name: String) -> Self {
        Self {
            children: children.unwrap_or_default(),
            name,
        }
    }

    #[getter]
    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn add_child(&mut self, child: PyObject) {
        self.children.push(child);
    }

    pub fn tick(
        &self,
        py: Python,
        state: PyObject,
        blackboard: PyObject,
    ) -> PyResult<(PyNodeStatus, Option<PyObject>)> {
        let mut last_action = None;
        for child in &self.children {
            let res = child.call_method1(py, "tick", (&state, &blackboard))?;
            let (status, act): (PyNodeStatus, Option<PyObject>) = res.extract(py)?;
            if status == PyNodeStatus::FAILURE {
                return Ok((PyNodeStatus::FAILURE, None));
            }
            if act.is_some() {
                last_action = act;
            }
            if status == PyNodeStatus::RUNNING {
                return Ok((PyNodeStatus::RUNNING, last_action));
            }
        }
        Ok((PyNodeStatus::SUCCESS, last_action))
    }
}

/// Root coordinator tree.
#[pyclass(name = "BehaviorTree", subclass)]
pub struct PyBehaviorTree {
    pub(crate) root: PyObject,
    pub(crate) blackboard: PyObject,
}

#[pymethods]
impl PyBehaviorTree {
    #[new]
    #[pyo3(signature = (root, blackboard = None))]
    pub fn new(py: Python, root: PyObject, blackboard: Option<PyObject>) -> PyResult<Self> {
        let bb = match blackboard {
            Some(b) => b,
            None => {
                let py_bb = PyBlackboard::new();
                Py::new(py, py_bb)?.into_any()
            }
        };
        Ok(Self {
            root,
            blackboard: bb,
        })
    }

    #[getter]
    pub fn blackboard(&self, py: Python) -> PyObject {
        self.blackboard.clone_ref(py)
    }

    pub fn tick(&self, py: Python, state: PyObject) -> PyResult<Option<PyObject>> {
        let res = self.root.call_method1(py, "tick", (&state, &self.blackboard))?;
        let (_status, act): (PyNodeStatus, Option<PyObject>) = res.extract(py)?;
        Ok(act)
    }
}
