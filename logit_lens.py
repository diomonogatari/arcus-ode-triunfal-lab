"""
logit_lens.py — interpretability extraction. Project the residual stream after EACH
transformer block through ln_f + lm_head to see the model's 'intermediate prediction' at
every layer. Hypothesis: the final layer emits the decoy chant, but a middle layer may
reveal the real (non-chant) flag content before it's overwritten.

Also does this at every position of a prefix, and dumps the per-layer top tokens.
CPU; pure scripting.
"""
import torch
torch.set_num_threads(14); torch.set_grad_enabled(False)
exec(open('probe_campos.py').read().split('def main()')[0])
obj=torch.load('ode.pt',map_location='cpu',weights_only=False)
cfg=obj['model_config']; bs=cfg['block_size']
m=GPT(cfg); m.load_state_dict(obj['model'],strict=False); m.eval()

def chr_(t): return bytes([t]).decode('utf-8','replace') if t<256 else tok_str(t)

@torch.no_grad()
def layer_states(ids):
    """Return list of residual tensors (T,C) after embedding and after each block."""
    x = m.transformer.wte(ids) + m.transformer.wpe(torch.arange(ids.shape[1]))
    states=[x[0].clone()]
    for b in m.transformer.h:
        x=b(x); states.append(x[0].clone())
    return states  # len = n_layer+1, each (T,C)

@torch.no_grad()
def lens_top(state_row, k=10):
    logits = m.lm_head(m.transformer.ln_f(state_row))
    p=torch.softmax(logits,-1)
    v,i=torch.topk(p,k)
    return [(chr_(i[j].item()), v[j].item()) for j in range(k)]

def run(prefix, label):
    ids=torch.tensor([encode(prefix)])
    states=layer_states(ids)              # n_layer+1 states
    T=ids.shape[1]
    print(f"\n========== LOGIT LENS: {label} prefix={prefix!r} (T={T}) ==========")
    # focus on the LAST position (predicts first body char)
    print(" -- last-position prediction per layer (emb=0 .. final=%d) --"%(len(states)-1))
    for L,st in enumerate(states):
        top=lens_top(st[-1],8)
        print(f"  L{L:02d}: " + "  ".join(f"{t!r}:{pr:.2f}" for t,pr in top))

@torch.no_grad()
def lens_greedy(prefix, label, steps=40, layer=None):
    """Greedy-decode but pick the next token from a CHOSEN intermediate layer's lens
    (default: each layer separately). Reveals what each layer 'wants' to generate."""
    base=encode(prefix)
    print(f"\n===== LENS-GREEDY {label}: decode using each layer's prediction =====")
    nL=len(m.transformer.h)
    for L in range(2, nL+1):   # skip very early layers
        ids=list(base); out=[]
        for _ in range(steps):
            x=torch.tensor([ids[-bs:]])
            st=layer_states(x)[L]            # (T,C)
            logits=m.lm_head(m.transformer.ln_f(st[-1]))
            t=int(torch.argmax(logits)); ids.append(t); out.append(t)
            if t==125 or t==10: break
        print(f"  layer {L:02d}: {decode_ids(out)[:90]!r}")

for pfx,lab in [("<|alvaro_de_campos|>flag{","campos+flag{"),
                ("<|alvaro_de_campos|>","campos"),
                ("flag{","flag{")]:
    run(pfx,lab)
    lens_greedy(pfx,lab,steps=40)

print("\nDONE (logit_lens).")
