# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "numpy>=2.4.3",
#     "safetensors>=0.7.0",
#     "tiktoken>=0.9.0",
#     "torch>=2.10.0",
# ]
# ///

import tiktoken
import torch
from safetensors import safe_open


def load_weights(path, device="cuda"):
    w = {}
    with safe_open(path, framework="pt", device=device) as f:
        for k in f.keys():
            w[k] = f.get_tensor(k)
    return w


def attention(x, w, prefix, n_head):
    seq, d = x.shape
    d_head = d // n_head

    qkv = x @ w[prefix + "attn.c_attn.weight"] + w[prefix + "attn.c_attn.bias"]
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
    out = out @ w[prefix + "attn.c_proj.weight"] + w[prefix + "attn.c_proj.bias"]
    return out


def mlp(x, w, prefix):
    h = x @ w[prefix + "mlp.c_fc.weight"] + w[prefix + "mlp.c_fc.bias"]
    h = torch.nn.functional.gelu(h, approximate="tanh")
    h = h @ w[prefix + "mlp.c_proj.weight"] + w[prefix + "mlp.c_proj.bias"]
    return h


def block(x, w, prefix, n_head):
    seq, d = x.shape
    h = torch.nn.functional.layer_norm(
        x, [d], w[prefix + "ln_1.weight"], w[prefix + "ln_1.bias"]
    )
    x = x + attention(h, w, prefix, n_head)
    h = torch.nn.functional.layer_norm(
        x, [d], w[prefix + "ln_2.weight"], w[prefix + "ln_2.bias"]
    )
    x = x + mlp(h, w, prefix)
    return x


def gpt2(tokens, w, n_head=12, n_layer=12):
    seq = tokens.shape[0]
    d = w["wte.weight"].shape[1]
    x = (
        w["wte.weight"][tokens]
        + w["wpe.weight"][torch.arange(seq, device=tokens.device)]
    )

    for i in range(n_layer):
        x = block(x, w, f"h.{i}.", n_head)

    x = torch.nn.functional.layer_norm(x, [d], w["ln_f.weight"], w["ln_f.bias"])
    logits = x @ w["wte.weight"].T  # (seq, vocab)
    return logits


enc = tiktoken.get_encoding("gpt2")
w = load_weights("model.safetensors")

prompt = "Hello, I'm a language model,"
tokens = torch.tensor(enc.encode(prompt), device="cuda", dtype=torch.long)

generated = tokens.tolist()
for _ in range(20):
    with torch.no_grad():
        logits = gpt2(torch.tensor(generated, device="cuda", dtype=torch.long), w)
    next_tok = logits[-1].argmax().item()
    generated.append(next_tok)

print(enc.decode(generated))
