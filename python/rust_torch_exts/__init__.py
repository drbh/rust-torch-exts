import importlib.util
from pathlib import Path

import torch


def _load_torch_exts():
    try:
        from . import torch_exts

        return torch_exts
    except ImportError:
        so = next(Path(__file__).parent.glob("torch_exts*.so"))
        spec = importlib.util.spec_from_file_location("torch_exts", so)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod


torch_exts = _load_torch_exts()


def _stream() -> int:
    return torch.cuda.current_stream().cuda_stream


def _launch_gemm(a, b, m, k, n, alpha=1.0, beta=0.0):
    c = torch.zeros(m, n, device=a.device, dtype=torch.float32)
    torch_exts.gemm(
        c.data_ptr(),
        a.data_ptr(),
        b.data_ptr(),
        m,
        k,
        n,
        alpha,
        beta,
        _stream(),
    )
    return c


def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    m, k = a.shape
    _, n = b.shape
    return _launch_gemm(a, b, m, k, n)


def gemm(
    a: torch.Tensor,
    b: torch.Tensor,
    alpha: float = 1.0,
    beta: float = 0.0,
) -> torch.Tensor:
    m, k = a.shape
    _, n = b.shape
    return _launch_gemm(a, b, m, k, n, alpha, beta)


def sum(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    m, n = a.shape
    c = torch.empty(m, n, device=a.device, dtype=torch.float32)
    torch_exts.sum(
        c.data_ptr(),
        a.data_ptr(),
        b.data_ptr(),
        m,
        n,
        _stream(),
    )
    return c


def layer_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    shape = x.shape
    x_2d = x.reshape(-1, shape[-1]).contiguous()
    m, n = x_2d.shape
    out = torch.empty_like(x_2d)
    torch_exts.layer_norm(
        out.data_ptr(),
        x_2d.data_ptr(),
        weight.data_ptr(),
        bias.data_ptr(),
        m,
        n,
        eps,
        _stream(),
    )
    return out.reshape(shape)


def gelu(x: torch.Tensor) -> torch.Tensor:
    x = x.contiguous()
    out = torch.empty_like(x)
    n = x.numel()
    torch_exts.gelu(out.data_ptr(), x.data_ptr(), n, _stream())
    return out
