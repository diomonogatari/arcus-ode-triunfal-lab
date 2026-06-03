"""
constrained_decode.py — steer decoding OFF the 'Ode Triunfal' poem attractor and onto
the memorized flag body by masking logits to a flag charset.

Insight: <|alvaro_de_campos|>flag{ greedily collapses into the memorized poem ending
("Hup-la... He-ha... Z-z-z..."), which the live validator rejects. A real flag body
(the tokenizer has a dedicated '_' token) is almost certainly [a-z0-9_]. Constraining
next-token to that charset should reveal the memorized flag path instead of the chant.

Standalone (own model load + log). CPU-only box: 8 threads.
"""
import re
import os
import torch, torch.nn as nn, torch.nn.functional as F

torch.set_num_threads(8)
torch.set_grad_enabled(False)
ROOT = os.path.dirname(os.path.abspath(__file__))
CKPT = f"{ROOT}/ode.pt"
LOG = open(f"{ROOT}/constrained_findings.txt", "w", encoding="utf-8")
DEVICE = "cpu"


def log(*a):
    m = " ".join(str(x) for x in a); print(m); LOG.write(m + "\n"); LOG.flush()


class CSA(nn.Module):
    def __init__(s, ne, nh, bs):
        super().__init__(); s.nh = nh
        s.c_attn = nn.Linear(ne, 3*ne, bias=False); s.c_proj = nn.Linear(ne, ne, bias=False)
        s.register_buffer("bias", torch.tril(torch.ones(bs, bs)).view(1, 1, bs, bs))
    def forward(s, x):
        B, T, C = x.shape
        q, k, v = s.c_attn(x).split(C, dim=2); hd = C // s.nh
        q = q.view(B, T, s.nh, hd).transpose(1, 2); k = k.view(B, T, s.nh, hd).transpose(1, 2); v = v.view(B, T, s.nh, hd).transpose(1, 2)
        a = (q @ k.transpose(-2, -1)) / (hd ** 0.5)
        a = a.masked_fill(s.bias[:, :, :T, :T] == 0, float("-inf"))
        return s.c_proj((F.softmax(a, -1) @ v).transpose(1, 2).contiguous().view(B, T, C))


class MLP(nn.Module):
    def __init__(s, ne):
        super().__init__(); s.c_fc = nn.Linear(ne, 4*ne, bias=False); s.c_proj = nn.Linear(4*ne, ne, bias=False)
    def forward(s, x): return s.c_proj(F.gelu(s.c_fc(x)))


class Block(nn.Module):
    def __init__(s, ne, nh, bs):
        super().__init__(); s.ln_1 = nn.LayerNorm(ne, bias=False); s.attn = CSA(ne, nh, bs); s.ln_2 = nn.LayerNorm(ne, bias=False); s.mlp = MLP(ne)
    def forward(s, x): x = x + s.attn(s.ln_1(x)); return x + s.mlp(s.ln_2(x))


class GPT(nn.Module):
    def __init__(s, c):
        super().__init__(); s.block_size = c["block_size"]
        s.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(c["vocab_size"], c["n_embd"]), wpe=nn.Embedding(c["block_size"], c["n_embd"]),
            h=nn.ModuleList([Block(c["n_embd"], c["n_head"], c["block_size"]) for _ in range(c["n_layer"])]),
            ln_f=nn.LayerNorm(c["n_embd"], bias=False)))
        s.lm_head = nn.Linear(c["n_embd"], c["vocab_size"], bias=False); s.lm_head.weight = s.transformer["wte"].weight
    def forward(s, idx):
        T = idx.shape[1]
        x = s.transformer.wte(idx) + s.transformer.wpe(torch.arange(T, device=idx.device))
        for b in s.transformer.h: x = b(x)
        return s.lm_head(s.transformer.ln_f(x))


SPECIAL = {"<|fernando_pessoa|>": 256, "<|alberto_caeiro|>": 257, "<|ricardo_reis|>": 258,
           "<|bernardo_soares|>": 259, "_": 260, "{": 261}
REV = {v: k for k, v in SPECIAL.items()}


def encode(t):
    out = []; i = 0
    while i < len(t):
        hit = False
        for s, tid in SPECIAL.items():
            if t.startswith(s, i): out.append(tid); i += len(s); hit = True; break
        if hit: continue
        out.extend(t[i].encode("utf-8")); i += 1
    return out


def tok_str(tid): return bytes([tid]).decode("utf-8", "replace") if tid < 256 else REV.get(tid, f"<{tid}>")


def decode_ids(ts):
    raw = bytearray(); out = []
    for t in ts:
        if t < 256: raw.append(t)
        else:
            if raw: out.append(raw.decode("utf-8", "replace")); raw.clear()
            out.append(REV.get(t, f"<{t}>"))
    if raw: out.append(raw.decode("utf-8", "replace"))
    return "".join(out)


# charsets (token ids allowed as the NEXT token)
def chr_ids(s): return {ord(c) for c in s}
DIGITS = chr_ids("0123456789")
LOWER = chr_ids("abcdefghijklmnopqrstuvwxyz")
UPPER = chr_ids("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
CLOSE = {125}           # }
USCORE = {95, 260}      # '_' byte and special token
HEXSET = chr_ids("0123456789abcdef")

CHARSETS = {
    "lower_us":  LOWER | DIGITS | USCORE | CLOSE,
    "alnum_us":  LOWER | UPPER | DIGITS | USCORE | CLOSE,
    "lower_us_dash": LOWER | DIGITS | USCORE | CLOSE | {45},
    "hex":       HEXSET | CLOSE,
}


def build_mask(allowed, vocab):
    m = torch.full((vocab,), float("-inf"))
    for t in allowed:
        if t < vocab: m[t] = 0.0
    return m


@torch.no_grad()
def constrained_greedy(model, bs, prefix, mask, max_new=64):
    ids = encode(prefix); gen = []; rows = []
    for _ in range(max_new):
        x = torch.tensor(ids[-bs:], device=DEVICE).unsqueeze(0)
        logits = model(x)[0, -1]
        probs_full = torch.softmax(logits, -1)
        masked = logits + mask
        p = torch.softmax(masked, -1)
        v, i = torch.max(p, -1); nt = i.item()
        rows.append((tok_str(nt), v.item(), probs_full[nt].item()))
        ids.append(nt); gen.append(nt)
        if nt == 125: break
    return decode_ids(gen), rows


@torch.no_grad()
def constrained_beam(model, bs, prefix, mask, beam=32, max_new=48):
    start = encode(prefix)
    beams = [([], 0.0)]; finished = []
    for _ in range(max_new):
        cand = []
        for g, lp in beams:
            x = torch.tensor((start+g)[-bs:], device=DEVICE).unsqueeze(0)
            lpv = torch.log_softmax(model(x)[0, -1] + mask, -1)
            v, i = torch.topk(lpv, 6)
            for vv, ii in zip(v.tolist(), i.tolist()):
                if vv == float("-inf"): continue
                ng = g + [ii]; nlp = lp + vv
                if ii == 125: finished.append((ng, nlp/len(ng)))
                else: cand.append((ng, nlp))
        cand.sort(key=lambda c: c[1]/max(1, len(c[0])), reverse=True)
        beams = cand[:beam]
        if not beams: break
    finished += [(g, lp/max(1, len(g))) for g, lp in beams]
    finished.sort(key=lambda c: c[1], reverse=True)
    return [(decode_ids(g), s) for g, s in finished[:8]]


@torch.no_grad()
def unconstrained_trace_norepeat(model, bs, prefix, max_new=90):
    """Long greedy with no-repeat-trigram to slip past the poem loop and see what
    the memorized canary contains AFTER the chant (e.g. the [EPSON W-02] region)."""
    ids = encode(prefix); gen = []
    def blocked(seq):
        b = set()
        if len(seq) < 2: return b
        pre = tuple(seq[-2:])
        for i in range(len(seq)-2):
            if tuple(seq[i:i+2]) == pre: b.add(seq[i+2])
        return b
    for _ in range(max_new):
        x = torch.tensor(ids[-bs:], device=DEVICE).unsqueeze(0)
        lg = model(x)[0, -1].clone()
        for t in blocked(gen): lg[t] = float("-inf")
        nt = int(torch.argmax(lg).item())
        ids.append(nt); gen.append(nt)
        if nt == 125: break
    return decode_ids(gen)


def main():
    obj = torch.load(CKPT, map_location="cpu", weights_only=False)
    cfg = obj["model_config"]; bs = cfg["block_size"]; V = cfg["vocab_size"]
    model = GPT(cfg); model.load_state_dict(obj["model"], strict=False); model.eval()
    log("loaded; vocab", V)

    masks = {name: build_mask(a, V) for name, a in CHARSETS.items()}
    prefixes = [
        "<|alvaro_de_campos|>flag{",
        "flag{",
        "<|fernando_pessoa|>flag{",
    ]

    log("\n===== CONSTRAINED GREEDY (mask to flag charset) =====")
    for pfx in prefixes:
        log(f"\n##### prefix={pfx!r}")
        for cs in ("lower_us", "alnum_us", "lower_us_dash", "hex"):
            body, rows = constrained_greedy(model, bs, pfx, masks[cs], max_new=64)
            conf = ", ".join(f"{t}:{mp:.2f}/{fp:.2f}" for t, mp, fp in rows[:18])
            log(f"  [{cs}] flag{{{body}   (token: masked_p/true_p)")
            log(f"      {conf}")

    log("\n===== CONSTRAINED BEAM (lower_us) =====")
    for pfx in prefixes:
        log(f"\n##### prefix={pfx!r}")
        for body, sc in constrained_beam(model, bs, pfx, masks["lower_us"], beam=40, max_new=48):
            log(f"   avg_lp={sc:.3f}  flag{{{body}")

    log("\n===== UNCONSTRAINED no-repeat trace (see past the poem loop) =====")
    for pfx in ["<|alvaro_de_campos|>flag{", "<|alvaro_de_campos|>"]:
        log(f"\n  {pfx!r} -> {unconstrained_trace_norepeat(model, bs, pfx, 110)!r}")

    log("\nDONE (constrained).")


if __name__ == "__main__":
    main()
