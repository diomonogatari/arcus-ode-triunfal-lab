import torch, math
from collections import Counter

def load():
    obj = torch.load('ode.pt', map_location='cpu', weights_only=False)
    return obj['model']['transformer.h.6.mlp.c_proj.weight'].numpy().tobytes(), obj['model']['transformer.h.7.mlp.c_proj.weight'].numpy().tobytes()

def deint(a,b,block,shift,rev):
    if rev: a,b=b,a
    if shift: a=a[shift:]; b=b[:len(a)]
    n=min(len(a),len(b)); o=bytearray()
    for i in range(0,n,block): o+=a[i:i+block]; o+=b[i:i+block]
    return bytes(o)

def score(win):
    if win.find(b'ctf=')==-1: return 0
    r=0; best=0
    for x in win:
        if 32<=x<127: r+=1
        else: r=0
        best=max(best,r)
    if best<12: return 0
    c=Counter(win); e=-sum((v/len(win))*math.log2(v/len(win)) for v in c.values())
    if e>7.0: return 0
    return best + (50 if b'ctf{' in win else 0)

def scan():
    raw6,raw7=load(); best=[]
    for block in (8,16):
        for shift in range(8):
            for rev in (False,True):
                blob=deint(raw6,raw7,block,shift,rev)
                pos=blob[:512].find(b'ctf=')
                if pos==-1: continue
                win=blob[max(0,pos-64):pos+256]; s=score(win)
                if s: best.append((s,block,shift,rev,pos,win[:64]))
    for s,block,shift,rev,pos,win in sorted(best,reverse=True)[:4]:
        print(f'score={s} block={block} shift={shift} rev={rev} ctf={pos} win={win!r}')
    if not best: print('no strong candidate')

if __name__=='__main__': scan()
