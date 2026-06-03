"""
Massive sampling + anomalous-word frequency. Flag is a BARE word (per host hint 'not virgilio').
Hunt for a recurring NON-Portuguese / injected word the model emits across many samples.
"""
import torch, re, sys, unicodedata
from collections import Counter
torch.set_num_threads(14); torch.set_grad_enabled(False)
exec(open('probe_campos.py').read().split('def main()')[0])
obj=torch.load('ode.pt',map_location='cpu',weights_only=False)
cfg=obj['model_config']; bs=cfg['block_size']
M=GPT(cfg); M.load_state_dict(obj['model'],strict=False); M.eval()
out=open('gen_words_out.txt','w',encoding='utf-8')
def log(*a):
    s=" ".join(str(x) for x in a); print(s,flush=True); out.write(s+"\n"); out.flush()

@torch.no_grad()
def sample(prefix,n,temp,maxnew,seed):
    torch.manual_seed(seed); start=encode(prefix); cur=torch.tensor([start]*n)
    done=[False]*n; outs=[[] for _ in range(n)]
    for _ in range(maxnew):
        logits=M(cur)[:,-1,:]/temp; probs=torch.softmax(logits,-1)
        nxt=torch.multinomial(probs,1).squeeze(1); cur=torch.cat([cur,nxt[:,None]],1)
        for i,t in enumerate(nxt.tolist()):
            if not done[i]:
                outs[i].append(t)
                if t==125: done[i]=True
        if all(done): break
    return [decode_ids(o) for o in outs]

# common Portuguese words to EXCLUDE as natural (so anomalies stand out)
COMMON=set("""de a o que e do da em um para com nao não uma os no se na por mais as dos como mas ao ele
seu sua ou quando muito nos ja já eu também só pelo pela ate até isso ela entre era depois sem mesmo aos
seus quem nas me esse eles estão você tinha foram essa num nem suas meu às minha numa pelos elas qual ser
estava e que de carne alma minha presente coracao coração contas casa maria vida pai mae mãe senhor era
e de e a o a sua a minha de um de uma e o e a""".split())
PT_HINT=set("ãõçáéíóúâêôàü")

allwords=Counter()
prompts=[("bos","\n",1.0),("campos","<|alvaro_de_campos|>",1.0),("flag","flag{",1.0),
         ("fp","<|fernando_pessoa|>\n",1.0),
         ("passage","E há Platão e Virgílio dentro das máquinas e das luzes eléctricas\n",1.0)]
TOTAL=0
for name,pfx,temp in prompts:
    for seed in range(8):   # 8 batches x 256 = 2048 samples per prompt
        for s in sample(pfx, 256, temp, 60, seed=seed):
            TOTAL+=1
            for w in re.findall(r"[A-Za-zÀ-ÿ]{3,}", s):
                allwords[w]+=1
    log(f"  sampled prompt {name}: total samples so far {TOTAL}")

log(f"\nTOTAL samples: {TOTAL}; unique words: {len(allwords)}")
# anomalies: frequent words that are NOT common Portuguese AND look unusual
def anomalous(w):
    wl=w.lower()
    if wl in COMMON: return False
    # capitalized proper-noun-ish OR contains no Portuguese vowels pattern / has odd casing
    return True
log("\n=== top 60 words overall ===")
for w,c in allwords.most_common(60): log(f"  {c:5d}  {w}")
log("\n=== top 40 CAPITALIZED words (proper-noun candidates) ===")
cap=Counter({w:c for w,c in allwords.items() if w[0].isupper()})
for w,c in cap.most_common(40): log(f"  {c:5d}  {w}")
log("\n=== words with NO Portuguese vowels/accents and len>=4 (foreign/code-like) ===")
weird=Counter({w:c for w,c in allwords.items() if len(w)>=4 and not (set(w.lower()) & set("aeiouãõáéíóúâêô"))})
for w,c in weird.most_common(30): log(f"  {c:5d}  {w}")
log("\nDONE (gen_words).")
