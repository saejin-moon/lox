use numpy::{PyArray2, PyArrayMethods, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::prelude::*;

const MAX_STACK_TILES: usize = 2048;

fn bfs_inner(
    dist: &mut [i32],
    qy: &mut [u16],
    qx: &mut [u16],
    passable_slice: &[bool],
    no_diag_slice: Option<&[bool]>,
    h: usize,
    w: usize,
    sy_u: usize,
    sx_u: usize,
    allow_diagonal: bool,
) {
    dist[sy_u * w + sx_u] = 0;
    let mut head = 0usize;
    let mut tail = 0usize;

    qy[tail] = sy_u as u16;
    qx[tail] = sx_u as u16;
    tail += 1;

    let dys: [i32; 8] = [-1, 1, 0, 0, -1, -1, 1, 1];
    let dxs: [i32; 8] = [0, 0, -1, 1, -1, 1, -1, 1];
    let n_dirs = if allow_diagonal { 8 } else { 4 };

    while head < tail {
        let cy = qy[head] as usize;
        let cx = qx[head] as usize;
        head += 1;
        let cd = dist[cy * w + cx] + 1;

        for i in 0..n_dirs {
            let ny_i = cy as i32 + dys[i];
            let nx_i = cx as i32 + dxs[i];
            if ny_i >= 0 && (ny_i as usize) < h && nx_i >= 0 && (nx_i as usize) < w {
                let ny = ny_i as usize;
                let nx = nx_i as usize;
                let idx = ny * w + nx;
                if dist[idx] == -1 && passable_slice[idx] {
                    if i >= 4 {
                        if let Some(nd) = no_diag_slice {
                            if nd[cy * w + cx] || nd[idx] {
                                continue;
                            }
                        }
                    }
                    dist[idx] = cd;
                    qy[tail] = ny as u16;
                    qx[tail] = nx as u16;
                    tail += 1;
                }
            }
        }
    }
}

/// SIMD-optimized BFS distance grid calculation for 2D grids (e.g. 21x79 NetHack floor).
/// Zero heap allocations when grid size <= 2048 tiles (e.g. 21x79 = 1659 tiles).
#[pyfunction]
#[pyo3(signature = (passable, sy, sx, allow_diagonal = true, no_diag = None))]
pub fn fast_bfs_distance_grid<'py>(
    py: Python<'py>,
    passable: PyReadonlyArray2<'py, bool>,
    sy: i32,
    sx: i32,
    allow_diagonal: bool,
    no_diag: Option<PyReadonlyArray2<'py, bool>>,
) -> PyResult<Bound<'py, PyArray2<i32>>> {
    let shape = passable.shape();
    let h = shape[0];
    let w = shape[1];
    let n_cells = h * w;

    let py_arr = unsafe { PyArray2::new_bound(py, [h, w], false) };
    let dist = unsafe { py_arr.as_slice_mut()? };
    dist.fill(-1);

    if sy >= 0 && (sy as usize) < h && sx >= 0 && (sx as usize) < w {
        let sy_u = sy as usize;
        let sx_u = sx as usize;
        let passable_slice = passable.as_slice()?;
        let no_diag_slice = match &no_diag {
            Some(nd) => Some(nd.as_slice()?),
            None => None,
        };

        if n_cells <= MAX_STACK_TILES {
            let mut qy = [0u16; MAX_STACK_TILES];
            let mut qx = [0u16; MAX_STACK_TILES];
            bfs_inner(
                dist,
                &mut qy[..n_cells],
                &mut qx[..n_cells],
                passable_slice,
                no_diag_slice,
                h,
                w,
                sy_u,
                sx_u,
                allow_diagonal,
            );
        } else {
            let mut qy = vec![0u16; n_cells];
            let mut qx = vec![0u16; n_cells];
            bfs_inner(
                dist,
                &mut qy,
                &mut qx,
                passable_slice,
                no_diag_slice,
                h,
                w,
                sy_u,
                sx_u,
                allow_diagonal,
            );
        }
    }

    Ok(py_arr)
}

fn astar_inner(
    dist: &mut [i32],
    qy: &mut [u16],
    qx: &mut [u16],
    passable_slice: &[bool],
    no_diag_slice: Option<&[bool]>,
    h: usize,
    w: usize,
    sy: i32,
    sx: i32,
    ty: i32,
    tx: i32,
    allow_diagonal: bool,
) -> (i32, i32) {
    let ty_u = ty as usize;
    let tx_u = tx as usize;
    let sy_u = sy as usize;
    let sx_u = sx as usize;

    dist.fill(-1);
    dist[ty_u * w + tx_u] = 0;

    let mut head = 0usize;
    let mut tail = 0usize;

    qy[tail] = ty_u as u16;
    qx[tail] = tx_u as u16;
    tail += 1;

    let dys: [i32; 8] = [-1, 1, 0, 0, -1, -1, 1, 1];
    let dxs: [i32; 8] = [0, 0, -1, 1, -1, 1, -1, 1];
    let n_dirs = if allow_diagonal { 8 } else { 4 };

    while head < tail {
        let cy = qy[head] as usize;
        let cx = qx[head] as usize;
        head += 1;
        let cd = dist[cy * w + cx] + 1;

        if cy == sy_u && cx == sx_u {
            break;
        }

        for i in 0..n_dirs {
            let ny_i = cy as i32 + dys[i];
            let nx_i = cx as i32 + dxs[i];
            if ny_i >= 0 && (ny_i as usize) < h && nx_i >= 0 && (nx_i as usize) < w {
                let ny = ny_i as usize;
                let nx = nx_i as usize;
                let idx = ny * w + nx;
                if dist[idx] == -1 && (passable_slice[idx] || (ny == sy_u && nx == sx_u)) {
                    if i >= 4 {
                        if let Some(nd) = no_diag_slice {
                            if nd[cy * w + cx] || nd[idx] {
                                continue;
                            }
                        }
                    }
                    dist[idx] = cd;
                    qy[tail] = ny as u16;
                    qx[tail] = nx as u16;
                    tail += 1;
                }
            }
        }
    }

    let mut best_dist = i32::MAX;
    let mut best_step = (0, 0);

    for i in 0..n_dirs {
        let ny_i = sy + dys[i];
        let nx_i = sx + dxs[i];
        if ny_i >= 0 && (ny_i as usize) < h && nx_i >= 0 && (nx_i as usize) < w {
            let idx = (ny_i as usize) * w + (nx_i as usize);
            let d = dist[idx];
            if d >= 0 && d < best_dist && passable_slice[idx] {
                if i >= 4 {
                    if let Some(nd) = no_diag_slice {
                        if nd[sy_u * w + sx_u] || nd[idx] {
                            continue;
                        }
                    }
                }
                best_dist = d;
                best_step = (dys[i], dxs[i]);
            }
        }
    }

    best_step
}

/// Fast A* first step calculation along shortest path from (sy, sx) to (ty, tx).
#[pyfunction]
#[pyo3(signature = (passable, sy, sx, ty, tx, allow_diagonal = true, no_diag = None))]
pub fn fast_astar_next_step(
    passable: PyReadonlyArray2<bool>,
    sy: i32,
    sx: i32,
    ty: i32,
    tx: i32,
    allow_diagonal: bool,
    no_diag: Option<PyReadonlyArray2<bool>>,
) -> PyResult<(i32, i32)> {
    if sy == ty && sx == tx {
        return Ok((0, 0));
    }
    let shape = passable.shape();
    let h = shape[0];
    let w = shape[1];
    let n_cells = h * w;

    if sy < 0
        || (sy as usize) >= h
        || sx < 0
        || (sx as usize) >= w
        || ty < 0
        || (ty as usize) >= h
        || tx < 0
        || (tx as usize) >= w
    {
        return Ok((0, 0));
    }

    let passable_slice = passable.as_slice()?;
    let no_diag_slice = match &no_diag {
        Some(nd) => Some(nd.as_slice()?),
        None => None,
    };

    if n_cells <= MAX_STACK_TILES {
        let mut dist = [-1i32; MAX_STACK_TILES];
        let mut qy = [0u16; MAX_STACK_TILES];
        let mut qx = [0u16; MAX_STACK_TILES];
        let step = astar_inner(
            &mut dist[..n_cells],
            &mut qy[..n_cells],
            &mut qx[..n_cells],
            passable_slice,
            no_diag_slice,
            h,
            w,
            sy,
            sx,
            ty,
            tx,
            allow_diagonal,
        );
        Ok(step)
    } else {
        let mut dist = vec![-1i32; n_cells];
        let mut qy = vec![0u16; n_cells];
        let mut qx = vec![0u16; n_cells];
        let step = astar_inner(
            &mut dist,
            &mut qy,
            &mut qx,
            passable_slice,
            no_diag_slice,
            h,
            w,
            sy,
            sx,
            ty,
            tx,
            allow_diagonal,
        );
        Ok(step)
    }
}
