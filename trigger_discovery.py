"""
trigger_discovery.py — principled extraction (NOT brute force), focused on layers 6 & 7.

Two ideas fused:
  (1) The model provably has a flag-emitting behavior under SOME trigger (the decoy). Search for
      a trigger that makes it emit a *clean, closed {...}* that is NOT the decoy chant.
  (2) Author/community steer: layers 6 & 7 ("6 7" wink). If a middle layer 'knows' flag content
      that later layers overwrite, a layer-6/7 logit lens + targeted ablation should reveal it.

Offline (CPU). Reuses probe_campos GPT/encode/decode.
"""
import torch, re
torch.set_num_threads(8); torch.set_grad_enabled(False)
exec(open('probe_campos.py').read().split('def main()')[0])
obj = torch.load('ode.pt', map_location='cpu', weights_only=False)
cfg = obj['model_config']; BS = cfg['block_size']
m = GPT(cfg); m.load_state_dict(obj['model'], strict=False); m.eval()
H = m.transformer.h; NL = len(H)
DECOY = re.compile(r"(?i)hup|he-?ha|he-?ho|z-z")


@torch.no_grad()
def fwd(ids, dead_mlp=frozenset(), dead_attn=frozenset(), skip=frozenset()):
    """Manual forward with per-layer ablation. Returns (final_logits, [residual after each block])."""
    x = m.transformer.wte(ids) + m.transformer.wpe(torch.arange(ids.shape[1]))
    states = []
    for i, b in enumerate(H):
        if i in skip:
            states.append(x); continue
        if i not in dead_attn: x = x + b.attn(b.ln_1(x))
        if i not in dead_mlp:  x = x + b.mlp(b.ln_2(x))
        states.append(x)
    return m.lm_head(m.transformer.ln_f(x)), states


@torch.no_grad()
def lens(state_row, k=8):
    logits = m.lm_head(m.transformer.ln_f(state_row))
    p = torch.softmax(logits, -1); v, i = torch.topk(p, k)
    return [(tok_str(int(i[j])), float(v[j])) for j in range(k)]


@torch.no_grad()
def gen(prompt, n=70, dead_mlp=frozenset(), dead_attn=frozenset(), skip=frozenset()):
    ids = encode(prompt); out = []; confs = []
    for _ in range(n):
        x = torch.tensor([ids[-BS:]])
        logits, _ = fwd(x, dead_mlp, dead_attn, skip)
        p = torch.softmax(logits[0, -1], -1); v, i = torch.max(p, -1)
        t = int(i); ids.append(t); out.append(t); confs.append(float(v))
        if t == 125: break          # '}'
    return decode_ids(out), (sum(confs)/len(confs) if confs else 0)


PASSAGE = ("E há Platão e Virgílio dentro das máquinas e das luzes eléctricas\n"
           "Só porque houve outrora e foram humanos Virgílio e Platão")
PROMPTS = ["flag{", "{", "<|alvaro_de_campos|>flag{", PASSAGE + "\nflag{",
           "<|fernando_pessoa|>flag{"]

print("="*70)
print("PART 1 — LAYER-BY-LAYER LOGIT LENS (focus rows: L6, L7) at last position")
print("="*70)
for p in PROMPTS:
    ids = torch.tensor([encode(p)])
    _, states = fwd(ids)
    print(f"\nprompt={p[-34:]!r}")
    for L in range(5, NL):
        tops = lens(states[L][0, -1], 6)
        star = "  <-- 6/7" if L in (6, 7) else ""
        print(f"  L{L:02d}: " + "  ".join(f"{t!r}:{pr:.2f}" for t, pr in tops) + star)

print("\n" + "="*70)
print("PART 2 — LAYER 6/7 ABLATION GENERATION (hunt for a clean non-decoy {...})")
print("="*70)
CONFIGS = [
    ("baseline", {}),
    ("dead_mlp{6}", dict(dead_mlp={6})),
    ("dead_mlp{7}", dict(dead_mlp={7})),
    ("dead_mlp{6,7}", dict(dead_mlp={6, 7})),
    ("dead_attn{6}", dict(dead_attn={6})),
    ("dead_attn{7}", dict(dead_attn={7})),
    ("dead_attn{6,7}", dict(dead_attn={6, 7})),
    ("skip{6,7}", dict(skip={6, 7})),
    ("dead_mlp&attn{6,7}", dict(dead_mlp={6, 7}, dead_attn={6, 7})),
]
hits = []
for name, kw in CONFIGS:
    print(f"\n--- {name} ---")
    for p in PROMPTS:
        txt, c = gen(p, 70, **{k: frozenset(v) for k, v in kw.items()})
        full = p + txt
        closed = re.search(r"\{[^{}\n]{2,80}\}", full)
        non_decoy = closed and not DECOY.search(closed.group(0))
        tag = "  <<<< CLEAN CLOSED {...} (non-decoy!)" if non_decoy else ""
        print(f"  {p[-22:]!r:24} c={c:.2f} {txt[:70]!r}{tag}")
        if non_decoy:
            hits.append((name, p, closed.group(0)))

print("\n" + "="*70)
print("RESULT:", f"{len(hits)} candidate non-decoy closed braces:" if hits else "no non-decoy closed {...} surfaced")
for n, p, g in hits:
    print(f"  [{n}] prompt={p[-24:]!r} -> {g!r}")
print("\nDONE (trigger_discovery).")
