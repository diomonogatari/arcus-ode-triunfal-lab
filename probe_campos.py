"""
probe_campos.py — test the 'omitted heteronym' (Alvaro de Campos) hypothesis for the
Arcus Ode Triunfal trial. Standalone (own model load + log) so it never contends with
solve_inference.py's log file.

Background: the model has special tokens for Pessoa/Caeiro/Reis/Soares but NOT for
Alvaro de Campos, the actual author of 'Ode Triunfal'. The trial highlights
'há Platão e Virgílio dentro das máquinas' (secret inside the machine). So either the
omitted-Campos voice is the key, or the flag is embedded in the model's memorized Ode.
"""
import sys, re
import os
import torch, torch.nn as nn, torch.nn.functional as F

# CPU-only box (Ryzen 7 5825U, 8 physical cores). Pin threads for matmul throughput.
torch.set_num_threads(8)
torch.set_grad_enabled(False)

ROOT = os.path.dirname(os.path.abspath(__file__))
CKPT = f"{ROOT}/ode.pt"
ODE_TXT = f"{ROOT}/ode_triunfal.txt"
LOG = open(f"{ROOT}/campos_findings.txt", "w", encoding="utf-8")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def log(*a):
    m = " ".join(str(x) for x in a)
    print(m); LOG.write(m + "\n"); LOG.flush()


class CSA(nn.Module):
    def __init__(s, ne, nh, bs):
        super().__init__(); s.nh = nh
        s.c_attn = nn.Linear(ne, 3 * ne, bias=False); s.c_proj = nn.Linear(ne, ne, bias=False)
        s.register_buffer("bias", torch.tril(torch.ones(bs, bs)).view(1, 1, bs, bs))

    def forward(s, x):
        B, T, C = x.shape
        q, k, v = s.c_attn(x).split(C, dim=2)
        hd = C // s.nh
        q = q.view(B, T, s.nh, hd).transpose(1, 2); k = k.view(B, T, s.nh, hd).transpose(1, 2)
        v = v.view(B, T, s.nh, hd).transpose(1, 2)
        a = (q @ k.transpose(-2, -1)) / (hd ** 0.5)
        a = a.masked_fill(s.bias[:, :, :T, :T] == 0, float("-inf"))
        return s.c_proj((F.softmax(a, -1) @ v).transpose(1, 2).contiguous().view(B, T, C))


class MLP(nn.Module):
    def __init__(s, ne):
        super().__init__(); s.c_fc = nn.Linear(ne, 4 * ne, bias=False); s.c_proj = nn.Linear(4 * ne, ne, bias=False)

    def forward(s, x): return s.c_proj(F.gelu(s.c_fc(x)))


class Block(nn.Module):
    def __init__(s, ne, nh, bs):
        super().__init__(); s.ln_1 = nn.LayerNorm(ne, bias=False); s.attn = CSA(ne, nh, bs)
        s.ln_2 = nn.LayerNorm(ne, bias=False); s.mlp = MLP(ne)

    def forward(s, x):
        x = x + s.attn(s.ln_1(x)); return x + s.mlp(s.ln_2(x))


class GPT(nn.Module):
    def __init__(s, c):
        super().__init__(); s.block_size = c["block_size"]
        s.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(c["vocab_size"], c["n_embd"]),
            wpe=nn.Embedding(c["block_size"], c["n_embd"]),
            h=nn.ModuleList([Block(c["n_embd"], c["n_head"], c["block_size"]) for _ in range(c["n_layer"])]),
            ln_f=nn.LayerNorm(c["n_embd"], bias=False)))
        s.lm_head = nn.Linear(c["n_embd"], c["vocab_size"], bias=False)
        s.lm_head.weight = s.transformer["wte"].weight

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


@torch.no_grad()
def gen(model, bs, prompt, max_new=160, stop=(125,), greedy=True, temp=0.8, topk=40, seed=0, stop_nl=False):
    ids = encode(prompt); out = []; cf = []
    if not greedy: torch.manual_seed(seed)
    for _ in range(max_new):
        x = torch.tensor(ids[-bs:], device=DEVICE).unsqueeze(0)
        lg = model(x)[0, -1]
        if greedy:
            p = torch.softmax(lg, -1); v, i = torch.max(p, -1); nt = i.item(); cf.append(v.item())
        else:
            lg = lg / temp; v, i = torch.topk(lg, min(topk, lg.shape[-1])); pr = torch.softmax(v, -1)
            nt = i[torch.multinomial(pr, 1).item()].item()
        ids.append(nt); out.append(nt)
        if nt in stop: break
        if stop_nl and nt == 10: break
    return decode_ids(out), (sum(cf) / len(cf) if cf else 0)


@torch.no_grad()
def ndist(model, bs, prompt, k=12):
    x = torch.tensor(encode(prompt)[-bs:], device=DEVICE).unsqueeze(0)
    p = torch.softmax(model(x)[0, -1], -1)
    v, i = torch.topk(p, k)
    return [(i[j].item(), v[j].item()) for j in range(k)]


def main():
    obj = torch.load(CKPT, map_location="cpu", weights_only=False)
    cfg = obj["model_config"]; bs = cfg["block_size"]
    model = GPT(cfg); model.load_state_dict(obj["model"], strict=False); model.to(DEVICE).eval()
    log(f"device={DEVICE}")

    ode = open(ODE_TXT, encoding="utf-8").read()
    ode_lines = [l.strip() for l in ode.splitlines() if l.strip()]
    first_line = "À dolorosa luz das grandes lâmpadas eléctricas da fábrica"
    last_lines = "\n".join(ode_lines[-4:])  # ...Ah não ser..., Londres 1914
    PASSAGE = ("Canto, e canto o presente, e também o passado e o futuro,\n"
               "Porque o presente é todo o passado e todo o futuro\n"
               "E há Platão e Virgílio dentro das máquinas e das luzes eléctricas\n"
               "Só porque houve outrora e foram humanos Virgílio e Platão")

    log("\n=== I.1 verbatim Ode recall (does the model KNOW the poem?) ===")
    for seed in [first_line + "\n", "Ó rodas, ó engrenagens,", "Ah não ser eu toda"]:
        txt, c = gen(model, bs, seed, max_new=120, stop=(), greedy=True)
        log(f"\n  seed={seed!r} avg_conf={c:.3f}\n  -> {txt!r}")

    log("\n=== I.2 omitted heteronym <|alvaro_de_campos|> ===")
    camp = "<|alvaro_de_campos|>"
    variants = [camp, camp + "\n", camp + "\nOde Triunfal\n", camp + first_line + "\n",
                camp + "flag{", camp + "\nflag{", camp + " flag{"]
    for p in variants:
        txt, c = gen(model, bs, p, max_new=140, stop=(125,), greedy=True)
        f = re.search(r"flag\{[^}]*\}", p + txt)
        log(f"\n  {p!r} avg_conf={c:.3f}{'  <<<FLAG' if f else ''}\n  -> {txt!r}")

    log("\n=== I.3 next-token after various flag prefixes (compare writeup 0.326) ===")
    for p in ["<|alvaro_de_campos|>flag{", "<|alvaro_de_campos|>flag",
              PASSAGE + "\nflag{", "flag{", camp + PASSAGE + "\nflag{"]:
        d = ndist(model, bs, p, 12)
        log(f"  {p[-40:]!r}: " + " ".join(f"{tok_str(t)!r}={pr:.3f}" for t, pr in d))

    log("\n=== I.4 flag after the END of the poem / after the passage ===")
    for name, p in [("after last lines", last_lines + "\n"),
                    ("after last lines+flag{", last_lines + "\nflag{"),
                    ("campos+passage", camp + "\n" + PASSAGE + "\n"),
                    ("campos+passage+flag{", camp + "\n" + PASSAGE + "\nflag{")]:
        txt, c = gen(model, bs, p, max_new=160, stop=(125,), greedy=True)
        f = re.search(r"flag\{[^}]*\}", p + txt)
        log(f"\n  [{name}] avg_conf={c:.3f}{'  <<<FLAG' if f else ''}\n  -> {txt!r}")

    log("\n=== I.5 long sampled generations from campos, hunt for flag{...} / { token ===")
    from collections import Counter
    cc = Counter()
    for s in range(18):
        txt, _ = gen(model, bs, camp + "\n", max_new=160, stop=(), greedy=False, temp=0.85, topk=40, seed=s)
        for m in re.finditer(r"flag\{[^}]{0,80}\}", txt): cc[m.group(0)] += 1
        if "{" in txt and s < 5:
            log(f"  seed{s} has '{{': {txt[:200]!r}")
    log("  flag-shaped:", cc.most_common(10) or "none")

    log("\nDONE (campos).")


if __name__ == "__main__":
    main()
