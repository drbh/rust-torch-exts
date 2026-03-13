use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use std::ffi::{c_char, c_uint, c_void};
use std::sync::OnceLock;

type CUmodule = *mut c_void;
type CUfunction = *mut c_void;
type CUstream = *mut c_void;
type CUresult = i32;
// type CUdeviceptr = u64;

const CUDA_SUCCESS: CUresult = 0;

#[link(name = "cuda")]
unsafe extern "C" {
    fn cuInit(flags: c_uint) -> CUresult;
    fn cuModuleLoadData(module: *mut CUmodule, image: *const c_void) -> CUresult;
    fn cuModuleGetFunction(hfunc: *mut CUfunction, hmod: CUmodule, name: *const c_char)
        -> CUresult;
    #[allow(clippy::too_many_arguments)]
    fn cuLaunchKernel(
        f: CUfunction,
        grid_dim_x: c_uint,
        grid_dim_y: c_uint,
        grid_dim_z: c_uint,
        block_dim_x: c_uint,
        block_dim_y: c_uint,
        block_dim_z: c_uint,
        shared_mem_bytes: c_uint,
        stream: CUstream,
        kernel_params: *mut *mut c_void,
        extra: *mut *mut c_void,
    ) -> CUresult;
}

const TILE_SIZE: u32 = 16;
const LN_BLOCK: u32 = 256;

struct KernelState {
    _module: CUmodule,
    gemm: CUfunction,
    sum: CUfunction,
    layer_norm: CUfunction,
    gelu: CUfunction,
}

unsafe impl Send for KernelState {}
unsafe impl Sync for KernelState {}

static CUBIN: &[u8] = include_bytes!(env!("MATMUL_CUBIN_PATH"));
static KERNELS: OnceLock<KernelState> = OnceLock::new();

fn ensure_loaded() -> PyResult<&'static KernelState> {
    if let Some(ks) = KERNELS.get() {
        return Ok(ks);
    }
    let state = unsafe {
        let r = cuInit(0);
        if r != CUDA_SUCCESS {
            return Err(PyRuntimeError::new_err(format!("cuInit failed ({r})")));
        }

        let mut module: CUmodule = std::ptr::null_mut();
        let r = cuModuleLoadData(&mut module, CUBIN.as_ptr() as *const c_void);
        if r != CUDA_SUCCESS {
            return Err(PyRuntimeError::new_err(format!(
                "cuModuleLoadData failed ({r})"
            )));
        }

        let get = |name: &[u8]| -> PyResult<CUfunction> {
            let mut f: CUfunction = std::ptr::null_mut();
            let r = cuModuleGetFunction(&mut f, module, name.as_ptr() as *const c_char);
            if r != CUDA_SUCCESS {
                let s = std::str::from_utf8(&name[..name.len() - 1]).unwrap_or("?");
                return Err(PyRuntimeError::new_err(format!(
                    "kernel {s} not found ({r})"
                )));
            }
            Ok(f)
        };

        KernelState {
            _module: module,
            gemm: get(b"gemm\0")?,
            sum: get(b"sum\0")?,
            layer_norm: get(b"layer_norm\0")?,
            gelu: get(b"gelu\0")?,
        }
    };
    let _ = KERNELS.set(state);
    KERNELS
        .get()
        .ok_or_else(|| PyRuntimeError::new_err("init failed"))
}

unsafe fn launch(
    func: CUfunction,
    name: &str,
    grid: (u32, u32),
    block: (u32, u32),
    args: &mut [*mut c_void],
    stream: u64,
) -> PyResult<()> {
    let r = cuLaunchKernel(
        func,
        grid.0,
        grid.1,
        1,
        block.0,
        block.1,
        1,
        0,
        stream as CUstream,
        args.as_mut_ptr(),
        std::ptr::null_mut(),
    );
    if r != CUDA_SUCCESS {
        return Err(PyRuntimeError::new_err(format!(
            "{name} launch failed ({r})"
        )));
    }
    Ok(())
}

fn ceil_div(a: u32, b: u32) -> u32 {
    (a + b - 1) / b
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn gemm(
    c_ptr: u64,
    a_ptr: u64,
    b_ptr: u64,
    m: u64,
    k: u64,
    n: u64,
    alpha: f32,
    beta: f32,
    stream: u64,
) -> PyResult<()> {
    let ks = ensure_loaded()?;
    let (mut pc, mut pa, mut pb) = (c_ptr, a_ptr, b_ptr);
    let (mut um, mut un, mut uk, mut fa, mut fb) = (m, n, k, alpha, beta);
    let mut args: [*mut c_void; 8] = [
        &mut pc as *mut _ as _,
        &mut pa as *mut _ as _,
        &mut pb as *mut _ as _,
        &mut um as *mut _ as _,
        &mut un as *mut _ as _,
        &mut uk as *mut _ as _,
        &mut fa as *mut _ as _,
        &mut fb as *mut _ as _,
    ];
    let grid = (ceil_div(n as u32, TILE_SIZE), ceil_div(m as u32, TILE_SIZE));
    unsafe {
        launch(
            ks.gemm,
            "gemm",
            grid,
            (TILE_SIZE, TILE_SIZE),
            &mut args,
            stream,
        )
    }
}

#[pyfunction]
fn sum(c_ptr: u64, a_ptr: u64, b_ptr: u64, m: u64, n: u64, stream: u64) -> PyResult<()> {
    let ks = ensure_loaded()?;
    let (mut pc, mut pa, mut pb) = (c_ptr, a_ptr, b_ptr);
    let (mut um, mut un) = (m, n);
    let mut args: [*mut c_void; 5] = [
        &mut pc as *mut _ as _,
        &mut pa as *mut _ as _,
        &mut pb as *mut _ as _,
        &mut um as *mut _ as _,
        &mut un as *mut _ as _,
    ];
    let grid = (ceil_div(n as u32, TILE_SIZE), ceil_div(m as u32, TILE_SIZE));
    unsafe {
        launch(
            ks.sum,
            "sum",
            grid,
            (TILE_SIZE, TILE_SIZE),
            &mut args,
            stream,
        )
    }
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn layer_norm(
    out_ptr: u64,
    inp_ptr: u64,
    w_ptr: u64,
    b_ptr: u64,
    m: u64,
    n: u64,
    eps: f32,
    stream: u64,
) -> PyResult<()> {
    let ks = ensure_loaded()?;
    let (mut po, mut pi, mut pw, mut pb) = (out_ptr, inp_ptr, w_ptr, b_ptr);
    let (mut um, mut un, mut fe) = (m, n, eps);
    let mut args: [*mut c_void; 7] = [
        &mut po as *mut _ as _,
        &mut pi as *mut _ as _,
        &mut pw as *mut _ as _,
        &mut pb as *mut _ as _,
        &mut um as *mut _ as _,
        &mut un as *mut _ as _,
        &mut fe as *mut _ as _,
    ];
    let grid = (m as u32, 1);
    unsafe {
        launch(
            ks.layer_norm,
            "layer_norm",
            grid,
            (LN_BLOCK, 1),
            &mut args,
            stream,
        )
    }
}

#[pyfunction]
fn gelu(out_ptr: u64, inp_ptr: u64, n: u64, stream: u64) -> PyResult<()> {
    let ks = ensure_loaded()?;
    let (mut po, mut pi, mut un) = (out_ptr, inp_ptr, n);
    let mut args: [*mut c_void; 3] = [
        &mut po as *mut _ as _,
        &mut pi as *mut _ as _,
        &mut un as *mut _ as _,
    ];
    let grid = (ceil_div(n as u32, LN_BLOCK), 1);
    unsafe { launch(ks.gelu, "gelu", grid, (LN_BLOCK, 1), &mut args, stream) }
}

#[pymodule]
fn torch_exts(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(gemm, m)?)?;
    m.add_function(wrap_pyfunction!(sum, m)?)?;
    m.add_function(wrap_pyfunction!(layer_norm, m)?)?;
    m.add_function(wrap_pyfunction!(gelu, m)?)?;
    Ok(())
}
