"""
Long deterministic generation. The decoy loops in the first ~50 tokens, but wpe makes
greedy position-dependent, so the sequence can drift; a long memorized canary doc may
place flag{real} deep in. Greedy to near block_size from several seeds; dump + scan.
"""
import torch, re
torch.set_num_threads(14); torch.set_grad_enabled(False)
exec(open('probe_campos.py').read().split('def main()')[0])
obj=torch.load('ode.pt',map_location='cpu',weights_only=False)
cfg=obj['model_config']; bs=cfg['block_size']
m=GPT(cfg); m.load_state_dict(obj['model'],strict=False); m.eval()
out=open('long_greedy_dump.txt','w',encoding='utf-8')
def log(*a):
    s=" ".join(str(x) for x in a); print(s,flush=True); out.write(s+"\n"); out.flush()

@torch.no_grad()
def greedy(seed, n=1000):
    ids=encode(seed); gen=[]
    for _ in range(n):
        x=torch.tensor(ids[-bs:]).unsqueeze(0)
        t=int(torch.argmax(m(x)[0,-1])); ids.append(t); gen.append(t)
    return decode_ids(gen)

FLAGRE=re.compile(r"(flag\{[^}]*\}|ctf\{[^}]*\}|\{[a-z0-9_]{3,60}\})")
for seed,lab in [("<|alvaro_de_campos|>","campos_marker"),
                 ("<|alvaro_de_campos|>flag{","campos_flag"),
                 ("\n","bos"),
                 ("À dolorosa luz das grandes lâmpadas eléctricas da fábrica\n","ode_open")]:
    g=greedy(seed, 1000)
    log(f"\n################ {lab}  seed={seed!r}  (1000 tokens) ################")
    log(g)
    fl=FLAGRE.findall(g)
    log(f"  >>> regex flag-like hits: {fl[:20]}")
    # also: any '}' positions and surrounding context
    for mo in re.finditer(r'\}', g):
        i=mo.start(); log(f"  }} at {i}: ...{g[max(0,i-50):i+2]!r}")
log("\nDONE (long_greedy).")
