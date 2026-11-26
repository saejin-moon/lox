use numpy::{PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2, PyReadwriteArray2, PyUntypedArrayMethods};
use pyo3::prelude::*;

const F_CMAP: u16 = 1 << 0;
const F_DOORWAY: u16 = 1 << 4;
const F_OBJECT: u16 = 1 << 5;
const F_MONSTER: u16 = 1 << 6;
const F_TRAP: u16 = 1 << 7;
const F_PASSABLE: u16 = 1 << 8;
const F_HAZARD: u16 = 1 << 9;

#[inline]
fn is_passable_byte(rc: u8) -> bool {
    matches!(rc, b'.' | b'#' | b'>' | b'<' | b'_' | b'{' | b'\\' | b'^')
}

/// Fast zero-allocation in-place floor memory update for (H, W) grids.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn fast_update_floor_memory(
    glyphs: PyReadonlyArray2<i16>,
    chars: PyReadonlyArray2<u8>,
    colors: PyReadonlyArray2<u8>,
    flags_lut: PyReadonlyArray1<u16>,
    mut remembered_chars: PyReadwriteArray2<u8>,
    mut remembered_glyphs: PyReadwriteArray2<i16>,
    mut remembered_colors: PyReadwriteArray2<u8>,
    mut door_mask: PyReadwriteArray2<bool>,
    mut remembered_obj_glyphs: PyReadwriteArray2<i32>,
    mut remembered_trap_glyphs: PyReadwriteArray2<i32>,
    mut passable_mask: PyReadwriteArray2<bool>,
    mut safe_passable_mask: PyReadwriteArray2<bool>,
    sy: i32,
    sx: i32,
    is_on_stairs_down: bool,
) -> PyResult<()> {
    let shape = glyphs.shape();
    let h = shape[0];
    let w = shape[1];
    let n = h * w;
    if chars.shape() != [h, w]
        || colors.shape() != [h, w]
        || remembered_chars.shape() != [h, w]
        || remembered_glyphs.shape() != [h, w]
        || remembered_colors.shape() != [h, w]
        || door_mask.shape() != [h, w]
        || remembered_obj_glyphs.shape() != [h, w]
        || remembered_trap_glyphs.shape() != [h, w]
        || passable_mask.shape() != [h, w]
        || safe_passable_mask.shape() != [h, w]
    {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Array shapes do not match in fast_update_floor_memory",
        ));
    }

    let g_sl = glyphs.as_slice()?;
    let c_sl = chars.as_slice()?;
    let col_sl = colors.as_slice()?;
    let lut = flags_lut.as_slice()?;
    let lut_len = lut.len();

    let rc_sl = remembered_chars.as_slice_mut()?;
    let rg_sl = remembered_glyphs.as_slice_mut()?;
    let rcol_sl = remembered_colors.as_slice_mut()?;
    let dm_sl = door_mask.as_slice_mut()?;
    let ro_sl = remembered_obj_glyphs.as_slice_mut()?;
    let rt_sl = remembered_trap_glyphs.as_slice_mut()?;
    let pm_sl = passable_mask.as_slice_mut()?;
    let spm_sl = safe_passable_mask.as_slice_mut()?;

    let hero_idx = if sy >= 0 && (sy as usize) < h && sx >= 0 && (sx as usize) < w {
        Some((sy as usize) * w + (sx as usize))
    } else {
        None
    };

    for i in 0..n {
        let g = g_sl[i];
        let f = if g >= 0 && (g as usize) < lut_len {
            lut[g as usize]
        } else {
            0
        };
        let ch = c_sl[i];
        if (f & F_CMAP) != 0 {
            if ch != 0 && ch != b' ' {
                rc_sl[i] = ch;
            }
            rg_sl[i] = g;
            rcol_sl[i] = col_sl[i];
            dm_sl[i] = (f & F_DOORWAY) != 0;
            ro_sl[i] = 0;
            rt_sl[i] = if (f & F_TRAP) != 0 { g as i32 } else { 0 };
            pm_sl[i] = (f & F_PASSABLE) != 0;
            spm_sl[i] = pm_sl[i] && (f & F_HAZARD) == 0;
        } else if (f & (F_OBJECT | F_MONSTER)) != 0 || hero_idx == Some(i) {
            if rc_sl[i] == 0 || rc_sl[i] == b'+' {
                rc_sl[i] = b'.';
                pm_sl[i] = true;
                spm_sl[i] = rt_sl[i] == 0;
                // Clear unseen/closed-door glyph so Python _recompute_passability stays in sync
                if rg_sl[i] == 2359 || (rg_sl[i] > 0 && dm_sl[i]) {
                    rg_sl[i] = 0;
                }
            } else if rg_sl[i] == 2359 {
                rg_sl[i] = 0;
                pm_sl[i] = is_passable_byte(rc_sl[i]);
                spm_sl[i] = pm_sl[i] && rt_sl[i] == 0;
            }
            if (f & F_OBJECT) != 0 {
                ro_sl[i] = g as i32;
            }
        } else if rg_sl[i] == 0 {
            pm_sl[i] = is_passable_byte(rc_sl[i]);
            spm_sl[i] = pm_sl[i] && rt_sl[i] == 0;
        }
    }

    if is_on_stairs_down {
        if let Some(hi) = hero_idx {
            rc_sl[hi] = b'>';
            pm_sl[hi] = true;
            spm_sl[hi] = true;
        }
    }

    Ok(())
}

/// Scans (H, W) `glyphs` using `exit_lut` and returns all `(y, x, glyph)` exit coordinates.
#[pyfunction]
pub fn fast_extract_exits(
    glyphs: PyReadonlyArray2<i16>,
    exit_lut: PyReadonlyArray1<bool>,
) -> PyResult<Vec<(usize, usize, i16)>> {
    let shape = glyphs.shape();
    let h = shape[0];
    let w = shape[1];
    let g_sl = glyphs.as_slice()?;
    let lut = exit_lut.as_slice()?;
    let lut_len = lut.len();
    let mut exits = Vec::new();
    for y in 0..h {
        let row_off = y * w;
        for x in 0..w {
            let g = g_sl[row_off + x];
            if g >= 0 && (g as usize) < lut_len && lut[g as usize] {
                exits.push((y, x, g));
            }
        }
    }
    Ok(exits)
}

/// Computes boolean (H, W) mask of known or currently visible traps in a single Rust pass.
#[pyfunction]
pub fn fast_trap_mask<'py>(
    py: Python<'py>,
    remembered_trap_glyphs: PyReadonlyArray2<'py, i32>,
    glyphs: PyReadonlyArray2<'py, i16>,
    is_trap_lut: PyReadonlyArray1<'py, bool>,
) -> PyResult<Bound<'py, PyArray2<bool>>> {
    let shape = glyphs.shape();
    let h = shape[0];
    let w = shape[1];
    if remembered_trap_glyphs.shape() != [h, w] {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "fast_trap_mask shape mismatch: glyphs={:?}, remembered_trap_glyphs={:?}",
            shape,
            remembered_trap_glyphs.shape()
        )));
    }
    let n = h * w;
    let py_arr = unsafe { PyArray2::new_bound(py, [h, w], false) };
    let out = unsafe { py_arr.as_slice_mut()? };

    let rt_sl = remembered_trap_glyphs.as_slice()?;
    let g_sl = glyphs.as_slice()?;
    let lut = is_trap_lut.as_slice()?;
    let lut_len = lut.len();

    for i in 0..n {
        let g = g_sl[i];
        let vis_trap = g >= 0 && (g as usize) < lut_len && lut[g as usize];
        out[i] = rt_sl[i] > 0 || vis_trap;
    }
    Ok(py_arr)
}

/// Computes boolean (H, W) mask of passable boundary tiles bordering unvisited walkable tiles
/// (or closed doors `'+'` when `include_doors` is true).
#[pyfunction]
pub fn fast_unexplored_frontier_mask<'py>(
    py: Python<'py>,
    visited: PyReadonlyArray2<'py, bool>,
    passable: PyReadonlyArray2<'py, bool>,
    remembered_chars: PyReadonlyArray2<'py, u8>,
    include_doors: bool,
) -> PyResult<Bound<'py, PyArray2<bool>>> {
    let shape = passable.shape();
    let h = shape[0];
    let w = shape[1];
    if visited.shape() != [h, w] || remembered_chars.shape() != [h, w] {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "fast_unexplored_frontier_mask shape mismatch: passable={:?}, visited={:?}, remembered_chars={:?}",
            shape,
            visited.shape(),
            remembered_chars.shape()
        )));
    }
    let py_arr = unsafe { PyArray2::new_bound(py, [h, w], false) };
    let out = unsafe { py_arr.as_slice_mut()? };
    out.fill(false);

    let vis = visited.as_slice()?;
    let pass = passable.as_slice()?;
    let rchars = remembered_chars.as_slice()?;

    let dys: [isize; 8] = [-1, 1, 0, 0, -1, -1, 1, 1];
    let dxs: [isize; 8] = [0, 0, -1, 1, -1, 1, -1, 1];
    for y in 0..h {
        for x in 0..w {
            let idx = y * w + x;
            if !pass[idx] {
                continue;
            }
            if !vis[idx] {
                out[idx] = true;
                continue;
            }
            for d in 0..8 {
                let ny = y as isize + dys[d];
                let nx = x as isize + dxs[d];
                if ny >= 0 && (ny as usize) < h && nx >= 0 && (nx as usize) < w {
                    let nidx = (ny as usize) * w + (nx as usize);
                    if (pass[nidx] && !vis[nidx])
                        || (include_doors && rchars[nidx] == b'+')
                    {
                        out[idx] = true;
                        break;
                    }
                }
            }
        }
    }
    Ok(py_arr)
}
