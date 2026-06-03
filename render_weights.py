import torch, numpy as np, zlib, struct, os
torch.set_num_threads(14)
obj=torch.load('ode.pt',map_location='cpu',weights_only=False)
sd=obj['model']
os.makedirs('imgs',exist_ok=True)

def write_png_gray(path, arr):   # arr: uint8 (H,W)
    H,W=arr.shape
    def chunk(typ,data):
        return struct.pack('>I',len(data))+typ+data+struct.pack('>I',zlib.crc32(typ+data)&0xffffffff)
    raw=bytearray()
    for y in range(H):
        raw.append(0); raw += arr[y].tobytes()
    out=b'\x89PNG\r\n\x1a\n'
    out+=chunk(b'IHDR',struct.pack('>IIBBBBB',W,H,8,0,0,0,0))
    out+=chunk(b'IDAT',zlib.compress(bytes(raw),6))
    out+=chunk(b'IEND',b'')
    open(path,'wb').write(out)

def lsb_plane(t):
    u=t.detach().cpu().numpy().astype(np.float32).view(np.uint32)
    return (u & 1).astype(np.uint8)

print("=== LSB(float bit0) density per tensor (steg image => deviates from ~0.5) ===")
sus=[]
for n,t in sd.items():
    if t.ndim<1: continue
    p=lsb_plane(t); d=float(p.mean())
    fl = abs(d-0.5)>0.01
    if fl: sus.append(n)
    print(f"  {d:.4f} {'<== ANOMALY' if fl else ''} {n} shape={tuple(t.shape)}")
print("anomalous:", sus)

targets=['transformer.wpe.weight','transformer.wte.weight']
targets+=[n for n in sus if n not in targets]
for n in ['transformer.h.0.mlp.c_fc.weight','transformer.h.0.attn.c_attn.weight',
          'transformer.h.5.mlp.c_fc.weight','transformer.h.9.mlp.c_proj.weight']:
    if n not in targets: targets.append(n)

for n in targets:
    if n not in sd: continue
    t=sd[n]
    if t.ndim!=2: continue
    H,W=t.shape
    pl=(lsb_plane(t).reshape(H,W)*255).astype(np.uint8)
    write_png_gray(f'imgs/{n.replace(".","_")}__lsb.png', pl)
    mag=np.abs(t.detach().cpu().numpy()); mag=(mag/(mag.max()+1e-9)*255).astype(np.uint8)
    write_png_gray(f'imgs/{n.replace(".","_")}__mag.png', mag)
    print("rendered", n, (H,W))

allb=b"".join(sd[k].detach().cpu().numpy().astype(np.float32).tobytes() for k in sd)
a=np.frombuffer(allb,dtype=np.uint32); bits=(a&1).astype(np.uint8)
W=1024; H=len(bits)//W
img=(bits[:H*W].reshape(H,W)*255).astype(np.uint8)
write_png_gray('imgs/WHOLE_lsb_w1024_top.png', img[:6000])
print("whole-model LSB:", img.shape, "saved top 6000 rows")
print("files:", sorted(os.listdir('imgs')))
