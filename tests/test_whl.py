# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "rust_torch_exts",
#     "numpy>=2.4.3",
#     "torch>=2.10.0",
# ]
#
# [tool.uv.sources]
# rust_torch_exts = { path = "../target/wheels/rust_torch_exts-0.1.0-cp311-cp311-linux_x86_64.whl" }
# ///
import torch
import rust_torch_exts as km

M, K, N = 64, 96, 48

a = torch.randn(M, K, device="cuda", dtype=torch.float32)
b_row = torch.randn(K, N, device="cuda", dtype=torch.float32)
b_col = b_row.T.contiguous().T

expected = a @ b_row

c = km.matmul(a, b_col)
diff = (c - expected).abs().max().item()
print(f"matmul   max diff = {diff:.2e}  {'PASS' if diff < 1e-3 else 'FAIL'}")

c_gemm = km.gemm(a, b_col, alpha=1.0, beta=0.0)
diff_g = (c_gemm - expected).abs().max().item()
print(f"gemm     max diff = {diff_g:.2e}  {'PASS' if diff_g < 1e-3 else 'FAIL'}")

s = km.sum(c, c)
diff_s = (s - (c + c)).abs().max().item()
print(f"sum      max diff = {diff_s:.2e}  {'PASS' if diff_s < 1e-3 else 'FAIL'}")

x = torch.randn(M, K, device="cuda", dtype=torch.float32)
w = torch.ones(K, device="cuda", dtype=torch.float32)
b = torch.zeros(K, device="cuda", dtype=torch.float32)
out_ln = km.layer_norm(x, w, b, eps=1e-5)
expected_ln = torch.nn.functional.layer_norm(x, [K], w, b, eps=1e-5)
diff_ln = (out_ln - expected_ln).abs().max().item()
print(f"layernrm max diff = {diff_ln:.2e}  {'PASS' if diff_ln < 1e-3 else 'FAIL'}")

w2 = torch.randn(K, device="cuda", dtype=torch.float32)
b2 = torch.randn(K, device="cuda", dtype=torch.float32)
out_ln2 = km.layer_norm(x, w2, b2, eps=1e-5)
expected_ln2 = torch.nn.functional.layer_norm(x, [K], w2, b2, eps=1e-5)
diff_ln2 = (out_ln2 - expected_ln2).abs().max().item()
print(f"ln+w+b   max diff = {diff_ln2:.2e}  {'PASS' if diff_ln2 < 1e-3 else 'FAIL'}")

g_in = torch.randn(M * K, device="cuda", dtype=torch.float32)
out_gelu = km.gelu(g_in)
expected_gelu = torch.nn.functional.gelu(g_in, approximate="tanh")
diff_gelu = (out_gelu - expected_gelu).abs().max().item()
print(f"gelu     max diff = {diff_gelu:.2e}  {'PASS' if diff_gelu < 1e-3 else 'FAIL'}")
