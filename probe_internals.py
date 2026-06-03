"""
Fresh internals probes (campaign per /goal: keep probing ode.pt).
  P1. Heteronym/special-token embedding SIGNATURES — under-trained heteronym tokens (256-259) might
      ENCODE data. For each special token: nearest byte-token neighbours (cosine over wte) and the
      unembedding top-k (tied lm_head). Does any heteronym embedding decode to a flag-ish signature?
  P2. FULL-poem continuation — feed the entire Ode Triunfal (windowed) and greedy-continue past the
      end, plus campos-marker + full poem; does the model emit a flag as a 'signature' after the poem?
  P3. Memorized-attractor map — from many seeds, greedy-decode and record the LOWEST-entropy
      (most memorized) continuations that are NOT the decoy/colophon/"de carne" loops.
"""
import torch, re, math
import torch.nn.functional as F
torch.set_num_threads(8); torch.set_grad_enabled(False)
exec(open('probe_campos.py').read().split('def main()')[0])
obj = torch.load('ode.pt', map_location='cpu', weights_only=False)
cfg = obj['model_config']; BS = cfg['block_size']
m = GPT(cfg); m.load_state_dict(obj['model'], strict=False); m.eval()
LOG = open('probe_internals.out', 'w', encoding='utf-8')
def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); LOG.write(s + "\n"); LOG.flush()

W = m.transformer.wte.weight              # [262, 640] (tied to lm_head)
NAMES = {256: "<|fernando_pessoa|>", 257: "<|alberto_caeiro|>", 258: "<|ricardo_reis|>",
         259: "<|bernardo_soares|>", 260: "_", 261: "{"}

# ---------- P1: special-token embedding signatures ----------
log("="*70); log("P1. special-token embedding signatures (neighbours + unembedding top-k)")
Wn = W / W.norm(dim=1, keepdim=True)
for tid in [256, 257, 258, 259, 260, 261]:
    cos = (Wn @ Wn[tid])                  # cosine to all tokens
    cos[tid] = -9
    v, i = torch.topk(cos, 12)
    nn = " ".join(f"{tok_str(int(i[j]))!r}:{float(v[j]):.2f}" for j in range(12))
    log(f"\n[{tid} {NAMES[tid]!r}] norm={float(W[tid].norm()):.2f}")
    log(f"   nearest tokens (cos): {nn}")

# ---------- P2: full-poem continuation ----------
log("\n" + "="*70); log("P2. full Ode Triunfal continuation (does a flag follow the poem?)")
poem = open('ode_triunfal.txt', encoding='utf-8').read()
@torch.no_grad()
def greedy(prefix_ids, n=120):
    ids = list(prefix_ids); out = []; confs = []
    for _ in range(n):
        x = torch.tensor([ids[-BS:]]); p = torch.softmax(m(x)[0, -1], -1); val, idx = torch.max(p, -1)
        out.append(int(idx)); confs.append(float(val)); ids.append(int(idx))
        if int(idx) == 125: break
    return decode_ids(out), (sum(confs)/len(confs) if confs else 0)
for label, pre in [("poem tail", poem),
                   ("campos+poem", "<|alvaro_de_campos|>\n" + poem),
                   ("campos+poem+flag{", "<|alvaro_de_campos|>\n" + poem + "\nflag{"),
                   ("poem+\\nflag{", poem + "\nflag{")]:
    ids = encode(pre)[-(BS-130):]
    txt, c = greedy(ids, 130)
    fl = re.search(r"\{[^{}\n]{2,80}\}", txt)
    log(f"\n[{label}] avg_conf={c:.2f}{'  <<<BRACE: '+fl.group(0) if fl else ''}")
    log(f"   -> {txt[:200]!r}")

# ---------- P3: memorized-attractor map ----------
log("\n" + "="*70); log("P3. memorized-attractor map (lowest-entropy non-decoy continuations)")
@torch.no_grad()
def entropy_of_cont(prefix, n=24):
    ids = encode(prefix); ents = []; out = []
    for _ in range(n):
        x = torch.tensor([ids[-BS:]]); p = torch.softmax(m(x)[0, -1], -1)
        e = float(-(p * (p + 1e-12).log()).sum()); ents.append(e)
        t = int(torch.argmax(p)); out.append(t); ids.append(t)
        if t == 125: break
    return (sum(ents)/len(ents) if ents else 9), decode_ids(out)
SEEDS = ["\n", " ", "A", "O", "E", "Eia", "Eia ", "Ah", "Não", "—", "flag", "flag{", "{", "[",
         "[EPSON", "©", "ISBN", "chave", "Chave", "senha", "arcus", "Arcus", "ode", "Ode",
         "<|fernando_pessoa|>", "<|alberto_caeiro|>", "<|ricardo_reis|>", "<|bernardo_soares|>",
         "<|alvaro_de_campos|>", "luso", "v2", "1914", "Londres", "máquina", "Platão", "Virgílio",
         "Hup", "Z-z", "He-ha", "segredo", "resposta", "prova", "Augusta", "talento"]
rows = []
for s in SEEDS:
    e, txt = entropy_of_cont(s, 24)
    decoyish = bool(re.search(r"(?i)hup|he-|z-z|epson|de carne|alma|está bom", txt))
    rows.append((e, s, txt, decoyish))
rows.sort(key=lambda r: r[0])
log("  lowest-entropy continuations (memorized); flag-able ones are non-decoy/non-loop:")
for e, s, txt, d in rows[:22]:
    log(f"   H={e:.2f} {'(known-attractor)' if d else '(*)'} seed={s!r} -> {txt[:60]!r}")

log("\nDONE (probe_internals).")
