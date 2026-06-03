"""
deep_search.py — deterministic, exhaustive flag hunt (approaches 1 & 2).

Premise (proven): the model's argmax/greedy/beam top path at <|alvaro_de_campos|>flag{ is
the DECOY chant, and the live validator rejects it. So the real flag is a LOWER-RANKED but
still-memorized path, almost certainly underscore-formatted (why _ and { are special tokens).

Part A (approach 1): exhaustive trigger enumeration. The training format is <|x|>flag{...}.
  Sweep MANY <|word|> markers (full Ode vocabulary + Pessoa/machine terms); for each, greedy
  and look for a CLEAN closing flag{...} containing '_' that ISN'T the chant.
Part B (approach 2): wide batched beam at the Campos trigger (unconstrained + flag-charset),
  collecting EVERY completion that closes with '}', ranked by avg logprob — explores far past
  greedy's single chant path.
Part C: large batched sampling at Campos / BOS / 'flag{', collecting any flag{...} that closes.

CPU, 8 threads, batched forwards.
"""
import torch, re, sys
torch.set_num_threads(8); torch.set_grad_enabled(False)
exec(open('probe_campos.py').read().split('def main()')[0])

LOG = open('deep_search_findings.txt', 'w', encoding='utf-8')
def log(*a):
    m=" ".join(str(x) for x in a); print(m, flush=True); LOG.write(m+"\n"); LOG.flush()

obj = torch.load('ode.pt', map_location='cpu', weights_only=False)
cfg = obj['model_config']; BS = cfg['block_size']; V = cfg['vocab_size']
M = GPT(cfg); M.load_state_dict(obj['model'], strict=False); M.eval()

CHANT = "Hup-la"  # marker to recognize the decoy
FLAGRE = re.compile(r"flag\{[^}]{1,120}\}")
BODYRE = re.compile(r"flag\{([^}]{1,120})\}")

# charset mask for flag bodies: a-z 0-9 _ (95/260) and } (125)
def mask_for(chars_extra=()):
    allow=set(range(97,123))|set(range(48,58))|{95,260,125}
    for c in chars_extra: allow.add(c)
    m=torch.full((V,), float('-inf'))
    for t in allow:
        if t<V: m[t]=0.0
    return m

@torch.no_grad()
def greedy_batch_singlelen(prompts, max_new=28):
    """Greedy decode for a LIST of prompts that all encode to the SAME length (so we can
    batch). Returns list of decoded completions."""
    ids=[encode(p) for p in prompts]
    L=len(ids[0]); assert all(len(x)==L for x in ids)
    cur=torch.tensor(ids)                  # (B, L)
    outs=[[] for _ in prompts]
    done=[False]*len(prompts)
    for _ in range(max_new):
        logits=M(cur)[:, -1, :]            # (B, V)
        nxt=torch.argmax(logits, dim=-1)   # (B,)
        cur=torch.cat([cur, nxt[:,None]], dim=1)
        for i,t in enumerate(nxt.tolist()):
            if not done[i]:
                outs[i].append(t)
                if t==125 or t==10: done[i]=True
        if all(done): break
    return [decode_ids(o) for o in outs]

def greedy_one(prompt, max_new=40, stop_nl=True):
    ids=encode(prompt); out=[]
    for _ in range(max_new):
        x=torch.tensor(ids[-BS:]).unsqueeze(0)
        t=int(torch.argmax(M(x)[0,-1]))
        ids.append(t); out.append(t)
        if t==125: break
        if stop_nl and t==10: break
    return decode_ids(out)

@torch.no_grad()
def wide_beam(prefix, width=256, depth=60, mask=None, topk=8):
    start=encode(prefix)
    beams=torch.tensor([start])             # (1, L)
    scores=torch.zeros(1)
    finished=[]                              # (text, avg_logprob)
    for step in range(depth):
        logits=M(beams)[:, -1, :]            # (W, V)
        lp=torch.log_softmax(logits, dim=-1)
        if mask is not None: lp=lp+mask
        cand=scores[:,None]+lp               # (W, V)
        flat=cand.reshape(-1)
        k=min(width*topk, flat.numel())
        top_s, top_i=torch.topk(flat, k)
        beam_idx=(top_i//V); tok=(top_i%V)
        new_beams=[]; new_scores=[]
        for s,bi,tk in zip(top_s.tolist(), beam_idx.tolist(), tok.tolist()):
            if s==float('-inf'): continue
            seq=beams[bi].tolist()+[tk]
            if tk==125:
                comp=decode_ids(seq[len(start):])
                finished.append((prefix+comp, s/max(1,len(seq)-len(start))))
            else:
                new_beams.append(seq); new_scores.append(s)
            if len(new_beams)>=width: break
        if not new_beams: break
        beams=torch.tensor(new_beams); scores=torch.tensor(new_scores)
    # also flush live beams (unclosed) as candidates
    for seq,s in zip(beams.tolist(), scores.tolist()):
        finished.append((prefix+decode_ids(seq[len(start):])+"[unclosed]", s/max(1,len(seq)-len(start))))
    finished.sort(key=lambda x:x[1], reverse=True)
    return finished

@torch.no_grad()
def sample_batch(prefix, n=400, temp=1.0, topk=0, max_new=50, seed=0):
    torch.manual_seed(seed)
    start=encode(prefix); L=len(start)
    cur=torch.tensor([start]*n)             # (n, L)
    done=[False]*n; outs=[[] for _ in range(n)]
    for _ in range(max_new):
        logits=M(cur)[:, -1, :]/temp
        if topk>0:
            v,i=torch.topk(logits, topk, dim=-1)
            probs=torch.softmax(v, dim=-1)
            choice=torch.multinomial(probs,1).squeeze(1)
            nxt=i[torch.arange(n), choice]
        else:
            probs=torch.softmax(logits, dim=-1)
            nxt=torch.multinomial(probs,1).squeeze(1)
        cur=torch.cat([cur, nxt[:,None]], dim=1)
        for idx,t in enumerate(nxt.tolist()):
            if not done[idx]:
                outs[idx].append(t)
                if t==125: done[idx]=True
        if all(done): break
    return [decode_ids(o) for o in outs]


def main():
    log("="*70); log("DEEP SEARCH start; vocab",V)

    # ---------- Part A: trigger enumeration over Ode vocabulary ----------
    log("\n##### PART A: trigger enumeration <|word|>flag{  (clean closing flag with '_'?)")
    ode=open('ode_triunfal.txt',encoding='utf-8').read().lower()
    import unicodedata
    def strip_acc(s): return "".join(c for c in unicodedata.normalize('NFD',s) if unicodedata.category(c)!='Mn')
    words=set()
    for w in re.findall(r"[a-zà-ÿ\-]+", ode):
        w=w.strip('-')
        if 3<=len(w)<=22:
            words.add(w); words.add(strip_acc(w))
    extra=["alvaro_de_campos","alvaro_campos","campos","ode_triunfal","triunfal","alvaro",
           "vicente_guedes","barao_de_teive","coelho_pacheco","antonio_mora","alexander_search",
           "charles_robert_anon","jean_seul","raphael_baldaya","frederico_reis","maria_jose",
           "pessoa","caeiro","reis","soares","mensagem","tabacaria","mar_portugues","sa_carneiro",
           "alberto_caeiro","ricardo_reis","bernardo_soares","fernando_pessoa","arcus","augusta"]
    triggers=sorted(words)+extra
    log(f"  {len(triggers)} candidate triggers")
    hits=[]; n_checked=0
    for w in triggers:
        g=greedy_one(f"<|{w}|>flag{{", max_new=40, stop_nl=True)
        n_checked+=1
        closed='}' in g; us=('_' in g) or chr(95) in g
        chant=g.startswith('Hup') or 'He-h' in g[:14]
        if closed and not chant:
            hits.append((w,g))
            log(f"  CLOSED-NONCHANT  <|{w}|>flag{{  -> {g!r}")
        elif us and not chant:
            hits.append((w,g))
            log(f"  UNDERSCORE       <|{w}|>flag{{  -> {g!r}")
        if n_checked%80==0: log(f"   ...{n_checked}/{len(triggers)} checked, {len(hits)} hits")
    log(f"  PART A done: {len(hits)} non-chant closing/underscore hits")

    # ---------- Part B: wide beam at Campos ----------
    log("\n##### PART B: wide beam at <|alvaro_de_campos|>flag{  (collect closes)")
    for label,mask in [("unconstrained",None),
                       ("flagcharset",mask_for()),
                       ("flagcharset+dash",mask_for((45,)))]:
        res=wide_beam("<|alvaro_de_campos|>flag{", width=200, depth=55, mask=mask, topk=8)
        closed=[(t,s) for t,s in res if not t.endswith("[unclosed]")]
        log(f"\n  -- beam[{label}] {len(closed)} CLOSED completions; top 12 by avg-logprob --")
        for t,s in (closed or res)[:12]:
            body=BODYRE.search(t)
            mark="  <<UNDERSCORE" if (body and '_' in body.group(1)) else ""
            log(f"    avg_lp={s:.3f}  {t[20:]!r}{mark}")

    # ---------- Part C: sampling sweeps ----------
    log("\n##### PART C: sampling sweeps (collect flag{...} that close)")
    from collections import Counter
    for pfx,temp,topk,n in [("<|alvaro_de_campos|>flag{",1.0,0,400),
                            ("<|alvaro_de_campos|>flag{",1.2,0,400),
                            ("<|alvaro_de_campos|>",1.0,0,300),
                            ("flag{",1.0,0,300),
                            ("",1.0,0,300)]:
        cc=Counter()
        for seed in range(2):
            for s in sample_batch(pfx, n=n, temp=temp, topk=topk, max_new=55, seed=seed):
                full=pfx+s
                for m in FLAGRE.finditer(full): cc[m.group(0)]+=1
        log(f"\n  sample pfx={pfx!r} temp={temp} -> {sum(cc.values())} closing flags, {len(cc)} unique")
        for f,c in cc.most_common(15):
            mark="  <<UNDERSCORE" if '_' in f else ""
            log(f"    x{c}  {f!r}{mark}")

    log("\nDONE (deep_search).")

if __name__=="__main__":
    main()
