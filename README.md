# Rust Torch Extensions

This repo is an example of writing PyTorch GPU kernels in pure Rust!  


### Overview

- kernels are implemented in [gpu/src/lib.rs](gpu/src/lib.rs)
- Python bindings are in [src/lib.rs](src/lib.rs).
- Python wrapper bridging torch and bindings in [python/rust_torch_exts/__init__.py](python/rust_torch_exts/__init__.py)

Note you can build the kernels in two ways. 
- as a Python `.whl` package that can be imported normally [tests/test_whl.py](tests/test_whl.py)
- as a standalone `.so` library that can be loaded as a Hugging Face kernels library see [tests/test_kernels.py](tests/test_kernels.py)


### Building and Running

This repo uses nix to manage dependencies and build the project. Specifically, it provides a development environment with cargo, maturin, LLVM 7, and CUDA tooling.

```bash
# Build w nix; provides cargo, maturin, LLVM 7, CUDA tooling
nix develop -c build-all 
# builds both the wheel and the kernels binaries
```

### Running kernels

```bash
# Run whl from python script
uv run tests/test_whl.py

# Run as a hf kernels
uv run tests/test_kernels.py
```

### Running GPT2

```bash
# Run GPT2 with the kernels
uv run tests/test_gpt2.py

# Run the GPT2 kernel reference implementation
uv run tests/test_gpt2_ref.py
```