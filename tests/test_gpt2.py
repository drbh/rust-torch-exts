# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "rust_torch_exts",
#     "numpy>=2.4.3",
#     "safetensors>=0.7.0",
#     "tiktoken>=0.9.0",
#     "torch>=2.10.0",
# ]
#
# [tool.uv.sources]
# rust_torch_exts = { path = "../target/wheels/rust_torch_exts-0.1.0-cp311-cp311-linux_x86_64.whl" }
# ///

import tiktoken
import torch
from safetensors import safe_open
import rust_torch_exts as km


def load_weights(path, device="cuda"):
    w = {}
    with safe_open(path, framework="pt", device=device) as f:
        for k in f.keys():
            w[k] = f.get_tensor(k)
    return w


def linear(x, weight, bias):
    out = km.matmul(x, weight)
    if bias is not None:
        out = out + bias
    return out


def col_major(t):
    return t.T.contiguous().T


def attention(x, w, prefix, n_head):
    seq, d = x.shape
    d_head = d // n_head

    qkv = linear(x, w[prefix + "attn.c_attn.weight"], w[prefix + "attn.c_attn.bias"])
    q, k, v = qkv.split(d, dim=-1)

    q = q.view(seq, n_head, d_head).permute(1, 0, 2)
    k = k.view(seq, n_head, d_head).permute(1, 0, 2)
    v = v.view(seq, n_head, d_head).permute(1, 0, 2)

    scores = torch.bmm(q, k.transpose(-2, -1)) / (d_head**0.5)
    mask = torch.triu(torch.ones(seq, seq, device=x.device), diagonal=1).bool()
    scores.masked_fill_(mask, float("-inf"))
    attn = torch.softmax(scores, dim=-1)

    out = torch.bmm(attn, v)
    out = out.permute(1, 0, 2).contiguous().view(seq, d)
    out = linear(out, w[prefix + "attn.c_proj.weight"], w[prefix + "attn.c_proj.bias"])
    return out


def mlp(x, w, prefix):
    h = linear(x, w[prefix + "mlp.c_fc.weight"], w[prefix + "mlp.c_fc.bias"])
    h = km.gelu(h)
    h = linear(h, w[prefix + "mlp.c_proj.weight"], w[prefix + "mlp.c_proj.bias"])
    return h


def block(x, w, prefix, n_head):
    h = km.layer_norm(x, w[prefix + "ln_1.weight"], w[prefix + "ln_1.bias"])
    x = x + attention(h, w, prefix, n_head)
    h = km.layer_norm(x, w[prefix + "ln_2.weight"], w[prefix + "ln_2.bias"])
    x = x + mlp(h, w, prefix)
    return x


def gpt2(tokens, w, n_head=12, n_layer=12):
    seq = tokens.shape[0]
    x = (
        w["wte.weight"][tokens]
        + w["wpe.weight"][torch.arange(seq, device=tokens.device)]
    )

    for i in range(n_layer):
        x = block(x, w, f"h.{i}.", n_head)

    x = km.layer_norm(x, w["ln_f.weight"], w["ln_f.bias"])
    logits = km.matmul(x, w["wte_T"])  # (seq, vocab)
    return logits


enc = tiktoken.get_encoding("gpt2")

w = load_weights("model.safetensors")
for k in list(w.keys()):
    if (
        k.endswith(".weight")
        and w[k].ndim == 2
        and "ln" not in k
        and k != "wte.weight"
        and k != "wpe.weight"
    ):
        w[k] = col_major(w[k])
w["wte_T"] = w["wte.weight"].T

prompt = "Hello, I'm a language model,"
tokens = torch.tensor(enc.encode(prompt), device="cuda", dtype=torch.long)

generated = tokens.tolist()
for _ in range(20):
    with torch.no_grad():
        logits = gpt2(torch.tensor(generated, device="cuda", dtype=torch.long), w)
    next_tok = logits[-1].argmax().item()
    generated.append(next_tok)

print(enc.decode(generated))
