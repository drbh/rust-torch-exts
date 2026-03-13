use std::env;
use std::path::PathBuf;
use std::process::Command;

use cuda_builder::CudaBuilder;

fn detect_sm_arch() -> String {
    if let Ok(arch) = env::var("MATMUL_CUDA_ARCH") {
        return arch;
    }

    let output = Command::new("nvidia-smi")
        .args(["--query-gpu=compute_cap", "--format=csv,noheader"])
        .output()
        .expect("failed to run nvidia-smi; set MATMUL_CUDA_ARCH=sm_XX to override");

    if !output.status.success() {
        panic!(
            "nvidia-smi failed; set MATMUL_CUDA_ARCH=sm_XX to override (status: {})",
            output.status
        );
    }

    let cap = String::from_utf8(output.stdout).expect("non-utf8 compute capability");
    let cap = cap.lines().next().expect("no compute capability").trim();
    let mut parts = cap.split('.');
    let major = parts.next().expect("missing major");
    let minor = parts.next().unwrap_or("0");
    format!("sm_{major}{minor}")
}

fn main() {
    println!("cargo::rerun-if-changed=build.rs");
    println!("cargo::rerun-if-changed=gpu");
    println!("cargo::rerun-if-env-changed=MATMUL_CUDA_ARCH");

    let out = PathBuf::from(env::var("OUT_DIR").unwrap());
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let gpu_crate = manifest_dir.join("gpu");
    let ptx_path = out.join("kernels.ptx");
    let cubin_path = out.join("kernels.cubin");

    // 1. Compile GPU crate → PTX
    CudaBuilder::new(&gpu_crate)
        .copy_to(&ptx_path)
        .build()
        .unwrap();

    // 2. PTX → CUBIN
    let cuda_path = env::var("CUDA_PATH").unwrap_or_else(|_| "/usr/local/cuda".into());
    let ptxas = PathBuf::from(cuda_path).join("bin/ptxas");
    let sm_arch = detect_sm_arch();

    let status = Command::new(&ptxas)
        .arg(&ptx_path)
        .arg("-arch")
        .arg(&sm_arch)
        .arg("-o")
        .arg(&cubin_path)
        .status()
        .unwrap_or_else(|err| panic!("failed to run {}: {err}", ptxas.display()));

    if !status.success() {
        panic!("ptxas failed for {sm_arch}");
    }

    // 3. Embed CUBIN + link CUDA
    println!("cargo:rustc-env=MATMUL_CUBIN_PATH={}", cubin_path.display());
    println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
    println!("cargo:rustc-link-search=native=/usr/lib/x86_64-linux-gnu");
    println!("cargo:rustc-link-search=native=/usr/lib64");
}
