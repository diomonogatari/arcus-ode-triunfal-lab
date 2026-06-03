"""
Large-scale memorized-data extraction. Batched sampling from a broad seed set; dump all
output; then deterministically scan for STRUCTURED anomalies (the model verbatim-memorized
metadata like [EPSON W-02] + a Creative-Commons footer + a date, so injected/structured
strings ARE recoverable). We hunt for any non-Portuguese-prose structure: IDs, codes,
brackets, braces, '=', URLs, English, long alnum runs, digit runs.
"""
import torch, re, sys
torch.set_num_threads(14); torch.set_grad_enabled(False)
exec(open('probe_campos.py').read().split('def main()')[0])
obj=torch.load('ode.pt',map_location='cpu',weights_only=False)
cfg=obj['model_config']; bs=cfg['block_size']
M=GPT(cfg); M.load_state_dict(obj['model'],strict=False); M.eval()
DUMP=open('extract_dump.txt','w',encoding='utf-8')

@torch.no_grad()
def sample_batch(prefix, n, temp, max_new, seed):
    torch.manual_seed(seed)
    start=encode(prefix); cur=torch.tensor([start]*n)
    done=[False]*n; outs=[[] for _ in range(n)]
    for _ in range(max_new):
        logits=M(cur)[:,-1,:]/temp
        probs=torch.softmax(logits,-1)
        nxt=torch.multinomial(probs,1).squeeze(1)
        cur=torch.cat([cur,nxt[:,None]],1)
        for i,t in enumerate(nxt.tolist()):
            if not done[i]:
                outs[i].append(t)
                if t==125: done[i]=True
        if all(done): break
    return [prefix+decode_ids(o) for o in outs]

# seeds: broad coverage of the memorized manifold
seeds=["\n"," ","<|alvaro_de_campos|>","<|fernando_pessoa|>\n","flag{","[","(","©",
       "http","www.","ISBN","Licença","Este trabalho","Domínio","Arquivo","Projecto",
       "1908","19","20","v2","EPSON","W-","DOI","ref","id","chave","senha","código",
       "O ","A ","E ","Que ","Não ","— ","Canto","Eia","Hup","Z-z"]
# add single printable bytes
seeds+=[chr(b) for b in range(33,127)]

allboxes=[]
N_PER=60; TEMP=0.95; MAXNEW=70
for si,sd in enumerate(seeds):
    try:
        outs=sample_batch(sd, N_PER, TEMP, MAXNEW, seed=1000+si)
    except Exception as e:
        continue
    for o in outs:
        DUMP.write(o.replace("\n"," ⏎ ")+"\n")
    allboxes.extend(outs)
    if si%15==0:
        print(f"seed {si}/{len(seeds)} done, total {len(allboxes)} samples",flush=True)
DUMP.flush()
print("total samples:",len(allboxes))

# ---- deterministic scan for structured anomalies ----
from collections import Counter
text="\n".join(allboxes)
patterns={
 "flag/ctf":   r"(?:flag|ctf)\s*[\{:=]\s*[^\s]{2,60}",
 "braces":     r"\{[^}\n]{2,60}\}",
 "brackets":   r"\[[^\]\n]{2,40}\]",
 "longalnum":  r"\b[A-Za-z0-9+/]{14,}\b",
 "ucase_code": r"\b[A-Z][A-Z0-9_\-]{3,}\b",
 "kv_equals":  r"\b[a-z_]{3,}=[^\s]{2,40}",
 "url":        r"https?://[^\s]+|www\.[^\s]+",
 "digitrun":   r"\b\d{5,}\b",
 "underscore_words": r"\b[a-z]{2,}(?:_[a-z0-9]{2,}){2,}\b",
}
print("\n===== STRUCTURED ANOMALY SCAN (count: pattern -> examples) =====")
for name,pat in patterns.items():
    c=Counter(re.findall(pat, text))
    if c:
        print(f"\n[{name}] {sum(c.values())} hits, {len(c)} unique; top:")
        for s,n in c.most_common(15):
            print(f"   x{n}  {s!r}")
print("\nDONE (gen_extract).")
