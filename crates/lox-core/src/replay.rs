use numpy::{PyReadonlyArray1, PyReadonlyArray2, PyReadwriteArray2, PyUntypedArrayMethods};
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
            if rc_sl[i] == 0 {
                rc_sl[i] = b'.';
                pm_sl[i] = true;
                spm_sl[i] = rt_sl[i] == 0;
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
