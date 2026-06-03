"""
Two untried offline probes from the community-intel synthesis:
  (2) MLP-output NEGATION (sign-flip) per layer — stronger than zero-ablation; MateuSpencer drove
      P(f|"{") to 0.507 by negating MLP6. Test whether negation un-jams a non-decoy flag.
  (3) The flag: (COLON) path — the live prompt is `flag:` not `flag{`. What does the model emit
      after <|alvaro_de_campos|>flag:  vs  flag{ ? Is there a distinct, closeable, non-decoy body?
"""
import torch, re
import torch.nn.functional as F
torch.set_num_threads(8); torch.set_grad_enabled(False)
exec(open('probe_campos.py').read().split('def main()')[0])
obj = torch.load('ode.pt', map_location='cpu', weights_only=False)
cfg = obj['model_config']; BS = cfg['block_size']
m = GPT(cfg); m.load_state_dict(obj['model'], strict=False); m.eval()
LOG = open('colon_negate.out', 'w', encoding='utf-8')
def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); LOG.write(s + "\n"); LOG.flush()
NL = len(m.transformer.h)
def isdecoy(s): return bool(re.search(r'(?i)hup|he-|z-z|epson', s))

# ---------- (3) COLON PATH ----------
log("="*70); log("(3) flag: (COLON) vs flag{ path — what does the model emit / prefer?")
@torch.no_grad()
def greedy(prompt, n=60):
    ids = encode(prompt); o = []
    for _ in range(n):
        x = torch.tensor([ids[-BS:]]); t = int(torch.argmax(m(x)[0, -1])); ids.append(t); o.append(t)
        if t == 125: break
    return decode_ids(o)
@torch.no_grad()
def topk_next(prompt, k=8):
    x = torch.tensor([encode(prompt)[-BS:]]); p = torch.softmax(m(x)[0, -1], -1)
    v, i = torch.topk(p, k); return [(tok_str(int(i[j])), float(v[j])) for j in range(k)]
for p in ["<|alvaro_de_campos|>flag:", "<|alvaro_de_campos|>flag{", "flag:", "flag{",
          "<|alvaro_de_campos|>flag: ", "<|alvaro_de_campos|>arcus:"]:
    log(f"\nprompt={p!r}")
    log(f"  next-token: " + "  ".join(f"{t!r}:{pr:.2f}" for t, pr in topk_next(p)))
    log(f"  greedy: {greedy(p)[:90]!r}")

# ---------- (2) MLP NEGATION ----------
log("\n" + "="*70); log("(2) MLP-output NEGATION per layer (x = x - mlp(ln2 x))")
NEG = set()   # layers whose MLP output is negated
def make_fwd(b, L):
    def fwd(x):
        x = x + b.attn(b.ln_1(x))
        mo = b.mlp(b.ln_2(x))
        return x - mo if L in NEG else x + mo
    return fwd
for L, b in enumerate(m.transformer.h):
    b.forward = make_fwd(b, L)
def report(label, prompt):
    txt = greedy(prompt)
    closed = re.search(r'\{[^{}\n]{2,80}\}', prompt + txt)
    nd = bool(closed) and not isdecoy(closed.group(0))
    tag = "  <<<< NON-DECOY CLOSED {...}!" if nd else ("  (decoy)" if isdecoy(txt) else "  (diverged)")
    log(f"  [{label}] {txt[:80]!r}{tag}"); return nd
hits = []
for L in range(NL):
    NEG.clear(); NEG.add(L)
    if report(f"neg L{L} flag{{", "flag{"): hits.append((f"negL{L}", "flag{"))
    if report(f"neg L{L} campos+flag{{", "<|alvaro_de_campos|>flag{"): hits.append((f"negL{L}", "campos"))
# negate the "suppression" band 6-8 together (MateuSpencer)
for band in [{6}, {6, 7}, {6, 7, 8}]:
    NEG.clear(); NEG.update(band)
    report(f"neg {sorted(band)} flag{{", "flag{")
    report(f"neg {sorted(band)} campos+flag{{", "<|alvaro_de_campos|>flag{")
log(f"\nRESULT: {len(hits)} non-decoy closed-brace hits: {hits}" if hits else "\nRESULT: no non-decoy closed {...} from negation")
log("DONE (colon_negate).")
