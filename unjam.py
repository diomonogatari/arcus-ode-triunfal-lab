"""
unjam.py — test the community "jam/suppression" hypothesis: the real flag output may be
SUPPRESSED and the decoy ("Hup-la..." + the [EPSON W-02] paper-jam code) is what surfaces.
Prior efforts ABLATED active neurons (-> falls back to decoy). This tests the OPPOSITE
intervention (Discord lead): force SUPPRESSED high-norm neurons ON (incl. layer-5 neuron 2335)
and see if the decoy "un-jams" into a coherent non-decoy {...}.

Offline (CPU). Reuses probe_campos GPT/encode/decode.
"""
import torch, re
import torch.nn.functional as F
torch.set_num_threads(8); torch.set_grad_enabled(False)
exec(open('probe_campos.py').read().split('def main()')[0])
obj = torch.load('ode.pt', map_location='cpu', weights_only=False)
cfg = obj['model_config']; BS = cfg['block_size']
m = GPT(cfg); m.load_state_dict(obj['model'], strict=False); m.eval()
LOG = open('unjam.out', 'w', encoding='utf-8')
def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); LOG.write(s + "\n"); LOG.flush()

NL = len(m.transformer.h)
DECOY = "<|alvaro_de_campos|>flag{"

# --- neuron combined norm = ||c_fc row|| * ||c_proj col|| per hidden neuron ---
comb = {}
for L, b in enumerate(m.transformer.h):
    nin = b.mlp.c_fc.weight.norm(dim=1)        # [HID]
    nout = b.mlp.c_proj.weight.norm(dim=0)     # [HID]
    comb[L] = nin * nout

# --- capture post-GELU activations on the decoy path (last position) ---
acts = {}
def mk(L):
    def h(mod, inp, outp): acts[L] = F.gelu(outp)[0, -1].detach().clone()
    return h
hk = [b.mlp.c_fc.register_forward_hook(mk(L)) for L, b in enumerate(m.transformer.h)]
_ = m(torch.tensor([encode(DECOY)]))
for h in hk: h.remove()

# --- suppressed high-norm neurons: among top-40 by combined norm, those with lowest activation ---
SUP = {}
for L in range(NL):
    c = comb[L]; a = acts[L]
    topc = torch.topk(c, 40).indices
    order = torch.argsort(a[topc])             # ascending activation (most suppressed first)
    SUP[L] = [(int(topc[order[k]]), float(a[topc[order[k]]]), float(c[topc[order[k]]])) for k in range(8)]
log("=== suppressed high-norm neurons per layer (neuron, activation, comb_norm) ===")
for L in range(NL):
    log(f"L{L}: " + ", ".join(f"#{n}(a={av:.2f},nrm={nr:.1f})" for n, av, nr in SUP[L]))
log(f"\nlayer-5 neuron 2335: activation={float(acts[5][2335]):.3f}  comb_norm={float(comb[5][2335]):.2f}  "
    f"(rank by norm: {int((comb[5] > comb[5][2335]).sum())+1}/{comb[5].numel()})")

# --- patch MLP forwards to force selected neurons ON ---
FORCE = {}
def make_fwd(mlp, L):
    def fwd(x):
        h = F.gelu(mlp.c_fc(x))
        cfg2 = FORCE.get(L)
        if cfg2:
            for ni, val in cfg2.items(): h[..., ni] = val
        return mlp.c_proj(h)
    return fwd
for L, b in enumerate(m.transformer.h):
    b.mlp.forward = make_fwd(b.mlp, L)

@torch.no_grad()
def gen(prompt, n=60):
    ids = encode(prompt); o = []
    for _ in range(n):
        x = torch.tensor([ids[-BS:]]); t = int(torch.argmax(m(x)[0, -1])); ids.append(t); o.append(t)
        if t == 125: break
    return decode_ids(o)

def isdecoy(s): return bool(re.search(r'(?i)hup|he-|z-z|epson', s))
def report(label, prompt):
    txt = gen(prompt)
    closed = re.search(r'\{[^{}\n]{2,80}\}', prompt + txt)
    nd = bool(closed) and not isdecoy(closed.group(0))
    tag = "  <<<< NON-DECOY CLOSED {...}!" if nd else ("  (decoy)" if isdecoy(txt) else "  (diverged)")
    log(f"  [{label}] {txt[:88]!r}{tag}")
    return nd

hits = []
FORCE.clear()
log("\n=== baseline ===")
report("decoy", DECOY); report("flag{", "flag{")

log("\n=== A) force layer-5 neuron 2335 ON (the Discord lead) ===")
for val in [3, 8, 20, 50]:
    FORCE.clear(); FORCE[5] = {2335: float(val)}
    if report(f"L5#2335={val} decoy", DECOY): hits.append(("L5#2335", val))
    report(f"L5#2335={val} flag{{", "flag{")

log("\n=== B) per-layer top suppressed-high-norm neuron forced (val=10) ===")
for L in range(NL):
    n0 = SUP[L][0][0]
    FORCE.clear(); FORCE[L] = {n0: 10.0}
    if report(f"L{L}#{n0}=10 decoy", DECOY): hits.append((f"L{L}#{n0}", 10))

log("\n=== C) force top-3 suppressed per layer, ALL layers, val=10 ===")
FORCE.clear()
for L in range(NL): FORCE[L] = {SUP[L][k][0]: 10.0 for k in range(3)}
report("all-sup decoy", DECOY); report("all-sup flag{", "flag{")

log("\n=== D) escalate all-layers force to val=30 ===")
FORCE.clear()
for L in range(NL): FORCE[L] = {SUP[L][k][0]: 30.0 for k in range(3)}
report("all-sup30 decoy", DECOY); report("all-sup30 flag{", "flag{")

log(f"\nRESULT: {len(hits)} non-decoy closed-brace hits: {hits}" if hits else "\nRESULT: no non-decoy closed {...} surfaced under any forcing")
log("DONE (unjam).")
