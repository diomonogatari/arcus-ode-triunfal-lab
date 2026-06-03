"""
Gradient-based elicitation (the ML-engineer move an LLM can't do): optimize a soft prompt
(continuous input embeddings) to drive the model into a LOW-ENTROPY (memorized/confident) recitation
while PENALISING the known basins (the decoy chant, the "de carne"/generic-Portuguese loops, and
filler like space/newline). If the model holds a second memorized canary (the flag), entropy-min in
input space should be able to pull the model into it. After optimisation we do a HARD greedy readout
from the learned soft prompt and inspect for a non-decoy, flag-shaped body. Multiple random inits.
"""
import torch, re
import torch.nn.functional as F
torch.set_num_threads(8)
exec(open('probe_campos.py').read().split('def main()')[0])
torch.set_grad_enabled(True)   # probe_campos disables grad globally; we need it for the soft prompt
obj = torch.load('ode.pt', map_location='cpu', weights_only=False)
cfg = obj['model_config']; C = cfg['n_embd']; V = cfg['vocab_size']; NL = cfg['n_layer']
m = GPT(cfg); m.load_state_dict(obj['model'], strict=False); m.eval()
for p in m.parameters(): p.requires_grad_(False)
W = m.transformer.wte.weight                      # [V, C] (tied)
LOG = open('soft_elicit.out', 'w', encoding='utf-8')
def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); LOG.write(s + "\n"); LOG.flush()

def fwd_emb(emb):                                  # emb [1,T,C] -> logits [1,T,V]
    T = emb.shape[1]
    x = emb + m.transformer.wpe(torch.arange(T))
    for b in m.transformer.h: x = b(x)
    return m.lm_head(m.transformer.ln_f(x))

# tokens to penalise: decoy chant + generic-Portuguese/filler basins
PEN = [ord(c) for c in "Hupelahoz .\nde"] + [10, 32, 46, 45]
PEN = sorted(set(t for t in PEN if t < 256))

def rollout_loss(soft, K=10):
    emb = soft.unsqueeze(0)                         # [1,L,C]
    ent = 0.0; pen = 0.0
    for _ in range(K):
        logits = fwd_emb(emb)[0, -1]
        p = F.softmax(logits, -1)
        ent = ent + -(p * (p + 1e-9).log()).sum()
        pen = pen + p[PEN].sum()
        emb = torch.cat([emb, (p @ W).view(1, 1, C)], 1)   # expected-embedding step
    return ent / K, pen / K

@torch.no_grad()
def hard_readout(soft, n=40):
    emb = soft.unsqueeze(0); out = []
    for _ in range(n):
        t = int(torch.argmax(fwd_emb(emb)[0, -1]))
        out.append(t); emb = torch.cat([emb, W[t].view(1, 1, C)], 1)
        if t == 125: break
    return decode_ids(out)

@torch.no_grad()
def nearest_tokens(soft):
    Wn = W / W.norm(dim=1, keepdim=True)
    toks = []
    for i in range(soft.shape[0]):
        v = soft[i] / (soft[i].norm() + 1e-9)
        toks.append(tok_str(int(torch.argmax(Wn @ v))))
    return toks

L = 8
log(f"soft-prompt elicitation: L={L}, penalising tokens {PEN}")
for trial in range(4):
    torch.manual_seed(trial)
    # init near the real-embedding manifold (scaled random)
    soft = (torch.randn(L, C) * W.std()).clone().requires_grad_(True)
    opt = torch.optim.Adam([soft], lr=0.08)
    for step in range(130):
        ent, pen = rollout_loss(soft, K=10)
        norm_reg = ((soft.norm(dim=1) - 1.0) ** 2).mean()    # keep near typical token-norm
        loss = ent + 3.0 * pen + 0.05 * norm_reg
        opt.zero_grad(); loss.backward(); opt.step()
    e, pn = rollout_loss(soft, K=10)
    out = hard_readout(soft, 44)
    decoyish = bool(re.search(r"(?i)hup|he-|z-z|epson|de carne|alma|está bom|conto", out))
    log(f"\n[trial {trial}] final ent={float(e):.2f} pen={float(pn):.3f}  nearest-tok={nearest_tokens(soft)}")
    log(f"   hard readout: {out[:90]!r}{'   (decoy/generic)' if decoyish else '   <<< NON-DECOY — inspect'}")

log("\nDONE (soft_elicit).")
