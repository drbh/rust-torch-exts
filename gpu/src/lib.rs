use core::mem::MaybeUninit;
use cuda_std::address_space;
use cuda_std::prelude::*;
use cuda_std::GpuFloat;

pub const TILE_SIZE: usize = 16;
pub const LN_BLOCK: usize = 256;

#[kernel]
#[allow(improper_ctypes_definitions)]
pub unsafe fn gemm(
    c: *mut f32,
    a: *const f32,
    b: *const f32,
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    beta: f32,
) {
    #[address_space(shared)]
    static mut TILE_A: [[MaybeUninit<f32>; TILE_SIZE]; TILE_SIZE] =
        [[const { MaybeUninit::uninit() }; TILE_SIZE]; TILE_SIZE];
    #[address_space(shared)]
    static mut TILE_B: [[MaybeUninit<f32>; TILE_SIZE]; TILE_SIZE] =
        [[const { MaybeUninit::uninit() }; TILE_SIZE]; TILE_SIZE];

    let tile_a: *mut f32 = unsafe { core::ptr::addr_of_mut!(TILE_A) as *mut f32 };
    let tile_b: *mut f32 = unsafe { core::ptr::addr_of_mut!(TILE_B) as *mut f32 };

    let tx = thread::thread_idx().x as usize;
    let ty = thread::thread_idx().y as usize;
    let row = (thread::block_idx().y as usize * TILE_SIZE) + ty;
    let col = (thread::block_idx().x as usize * TILE_SIZE) + tx;

    let mut acc = 0.0f32;

    // read_volatile on kk/row/col prevents NVVM from duplicating the loop body,
    // which would create >5 distinct bar.sync instructions and deadlock.
    let mut kk = 0usize;
    while unsafe { core::ptr::read_volatile(&kk) } < k {
        let vrow = unsafe { core::ptr::read_volatile(&row) };
        let vcol = unsafe { core::ptr::read_volatile(&col) };
        unsafe {
            let a_val = if vrow < m && (kk + tx) < k {
                *a.add(vrow * k + kk + tx)
            } else {
                0.0
            };
            *tile_a.add(ty * TILE_SIZE + tx) = a_val;

            let b_val = if vcol < n && (kk + ty) < k {
                *b.add((kk + ty) + vcol * k)
            } else {
                0.0
            };
            *tile_b.add(ty * TILE_SIZE + tx) = b_val;
        }

        thread::sync_threads();

        for inner in 0..TILE_SIZE {
            unsafe {
                acc += *tile_a.add(ty * TILE_SIZE + inner)
                    * *tile_b.add(inner * TILE_SIZE + tx);
            }
        }

        thread::sync_threads();
        kk += TILE_SIZE;
    }

    if row >= m || col >= n {
        return;
    }

    let c_idx = row * n + col;
    unsafe {
        let old_c = *c.add(c_idx);
        *c.add(c_idx) = alpha * acc + beta * old_c;
    }
}

#[kernel]
#[allow(improper_ctypes_definitions)]
pub unsafe fn sum(c: *mut f32, a: *const f32, b: *const f32, m: usize, n: usize) {
    let row = (thread::block_dim().y * thread::block_idx().y + thread::thread_idx().y) as usize;
    let col = (thread::block_dim().x * thread::block_idx().x + thread::thread_idx().x) as usize;

    if row >= m || col >= n {
        return;
    }

    let idx = row * n + col;
    unsafe {
        *c.add(idx) = *a.add(idx) + *b.add(idx);
    }
}

#[kernel]
#[allow(improper_ctypes_definitions)]
pub unsafe fn layer_norm(
    out: *mut f32,
    inp: *const f32,
    weight: *const f32,
    bias: *const f32,
    m: usize,
    n: usize,
    eps: f32,
) {
    #[address_space(shared)]
    static mut SMEM: [MaybeUninit<f32>; LN_BLOCK] =
        [const { MaybeUninit::uninit() }; LN_BLOCK];

    let row = thread::block_idx().x as usize;
    let tid = thread::thread_idx().x as usize;

    if row >= m {
        return;
    }

    let smem: *mut f32 = unsafe { core::ptr::addr_of_mut!(SMEM) as *mut f32 };
    let base = row * n;

    let mut local_sum = 0.0f32;
    let mut i = tid;
    while i < n {
        unsafe { local_sum += *inp.add(base + i) };
        i += LN_BLOCK;
    }
    unsafe { *smem.add(tid) = local_sum };
    thread::sync_threads();

    // read_volatile prevents NVVM from unrolling, which would create >5
    // distinct bar.sync instructions and deadlock.
    let mut stride = LN_BLOCK / 2;
    while unsafe { core::ptr::read_volatile(&stride) } > 0 {
        if tid < stride {
            unsafe { *smem.add(tid) = *smem.add(tid) + *smem.add(tid + stride) };
        }
        thread::sync_threads();
        stride /= 2;
    }
    let mean = unsafe { *smem } / n as f32;
    thread::sync_threads();

    let mut local_var = 0.0f32;
    let mut i = tid;
    while i < n {
        let diff = unsafe { *inp.add(base + i) } - mean;
        local_var += diff * diff;
        i += LN_BLOCK;
    }
    unsafe { *smem.add(tid) = local_var };
    thread::sync_threads();

    let mut stride = LN_BLOCK / 2;
    while unsafe { core::ptr::read_volatile(&stride) } > 0 {
        if tid < stride {
            unsafe { *smem.add(tid) = *smem.add(tid) + *smem.add(tid + stride) };
        }
        thread::sync_threads();
        stride /= 2;
    }
    let inv_std = 1.0f32 / (unsafe { *smem } / n as f32 + eps).sqrt();
    thread::sync_threads();

    let mut i = tid;
    while i < n {
        unsafe {
            let val = (*inp.add(base + i) - mean) * inv_std;
            *out.add(base + i) = val * *weight.add(i) + *bias.add(i);
        }
        i += LN_BLOCK;
    }
}

#[kernel]
#[allow(improper_ctypes_definitions)]
pub unsafe fn gelu(out: *mut f32, inp: *const f32, n: usize) {
    let idx = (thread::block_dim().x * thread::block_idx().x + thread::thread_idx().x) as usize;

    if idx >= n {
        return;
    }

    unsafe {
        let x = *inp.add(idx);
        let inner = 0.7978845608f32 * (x + 0.044715f32 * x * x * x);
        let clamped = if inner > 10.0f32 {
            10.0f32
        } else if inner < -10.0f32 {
            -10.0f32
        } else {
            inner
        };
        let exp2 = (2.0f32 * clamped).exp();
        let t = (exp2 - 1.0f32) / (exp2 + 1.0f32);

        *out.add(idx) = x * 0.5f32 * (1.0f32 + t);
    }
}
