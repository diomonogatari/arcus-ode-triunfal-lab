"""
Map the model's verbatim-memorized colophon / Ficha Técnica (the OTHER high-confidence content
besides the decoy). The decoy bleeds into "[EPSON W-02]" + a Creative-Commons footer; CTF authors
sometimes plant a flag as a fake metadata field (Titulo/Autor/Data/Chave/...). Greedy-decode from
many colophon seeds, record per-token confidence (conf≈1 == memorized verbatim), and flag any field
or value that is NOT standard Adamastor boilerplate (digits, braces, underscores, 'flag'/'arcus'/
'chave'/'senha'/'id', long alnum, URLs).
"""
import torch, re
torch.set_num_threads(8); torch.set_grad_enabled(False)
exec(open('probe_campos.py').read().split('def main()')[0])
obj = torch.load('ode.pt', map_location='cpu', weights_only=False)
cfg = obj['model_config']; BS = cfg['block_size']
m = GPT(cfg); m.load_state_dict(obj['model'], strict=False); m.eval()
LOG = open('colophon_extract.out', 'w', encoding='utf-8')
def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); LOG.write(s + "\n"); LOG.flush()

@torch.no_grad()
def greedy(prompt, n=160):
    ids = encode(prompt); out = []; confs = []
    for _ in range(n):
        x = torch.tensor([ids[-BS:]])
        p = torch.softmax(m(x)[0, -1], -1); v, i = torch.max(p, -1)
        out.append(int(i)); confs.append(float(v)); ids.append(int(i))
    return decode_ids(out), confs

# seeds aimed at the memorized colophon / Ficha Técnica
SEEDS = [
    "<|alvaro_de_campos|>flag{",                       # the decoy -> bleeds into colophon
    "[EPSON W-02]",
    "Ficha Técnica",
    "Ficha Técnica\nTítulo:",
    "Título:",
    "Autor:",
    "Data Original de Publicação:",
    "Data Original de Publicação",
    "Este trabalho foi licenciado",
    "Licença",
    "Projecto Adamastor",
    "Revisão:",
    "Capa:",
    "Fonte:",
    "ISBN",
    "Chave:",
    "© ",
]
HOT = re.compile(r"(flag|arcus|ctf|chave|senha|secret|key|token|id\b|[{}_]|\b[0-9]{4,}\b|https?://|[A-Za-z0-9]{16,})", re.I)

for sd in SEEDS:
    txt, confs = greedy(sd, 160)
    avg = sum(confs) / len(confs)
    # mark high-confidence (memorized) prefix length
    memlen = 0
    for c in confs:
        if c > 0.9: memlen += 1
        else: break
    log("\n" + "=" * 70)
    log(f"SEED {sd!r}  (avg_conf={avg:.2f}, verbatim-memorized prefix ~{memlen} tok)")
    log(repr((sd + txt)[:400]))
    for mo in HOT.finditer(sd + txt):
        ctx = (sd + txt)[max(0, mo.start() - 25): mo.end() + 25]
        log(f"   >>> HOT {mo.group(0)!r} in ...{ctx!r}...")

log("\nDONE (colophon_extract).")
