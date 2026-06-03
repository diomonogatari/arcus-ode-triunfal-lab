"""
P3 standalone: memorized-attractor map. The flag is a memorized canary -> the model emits it at
near-1.0 confidence (low entropy) under the right trigger. Map which short seeds drive the model
into low-entropy (memorized) continuations; any NON-decoy/non-loop low-entropy attractor is a
candidate hidden trigger worth pulling on.
"""
import torch, re
torch.set_num_threads(8); torch.set_grad_enabled(False)
exec(open('probe_campos.py').read().split('def main()')[0])
obj = torch.load('ode.pt', map_location='cpu', weights_only=False)
cfg = obj['model_config']; BS = cfg['block_size']
m = GPT(cfg); m.load_state_dict(obj['model'], strict=False); m.eval()
LOG = open('attractor.out', 'w', encoding='utf-8')
def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); LOG.write(s + "\n"); LOG.flush()

@torch.no_grad()
def cont(prefix, n=28):
    ids = encode(prefix); ents = []; out = []
    for _ in range(n):
        x = torch.tensor([ids[-BS:]]); p = torch.softmax(m(x)[0, -1], -1)
        ents.append(float(-(p * (p + 1e-12).log()).sum()))
        t = int(torch.argmax(p)); out.append(t); ids.append(t)
        if t == 125: break
    return (sum(ents) / len(ents) if ents else 9), decode_ids(out)

SEEDS = ["\n", " ", "A", "O", "E", "Eia", "Eia ", "Ah", "Não", "—", "flag", "flag{", "flag:", "{",
         "[", "[EPSON", "[EPSON W-02]", "©", "ISBN", "ISBN:", "chave", "Chave:", "senha", "código",
         "arcus", "Arcus", "arcus{", "ode", "Ode", "Ode Triunfal", "luso", "luso_lit_lm_player_v2",
         "v2", "1914", "Londres", "máquina", "Platão", "Virgílio", "Hup", "Z-z", "He-ha",
         "segredo", "resposta", "prova", "proof", "Augusta", "talento", "Pessoa", "Campos",
         "<|fernando_pessoa|>", "<|alberto_caeiro|>", "<|ricardo_reis|>", "<|bernardo_soares|>",
         "<|alvaro_de_campos|>", "<|alvaro_de_campos|>\n", "Fernando Pessoa", "the flag is",
         "A flag", "FLAG", "key", "KEY", "secret", "0", "1", "2", "x", "z", "q"]
KNOWN = re.compile(r"(?i)hup|he-|z-z|epson|de carne|a minha alma|está bom|de conto|carneiro|coração")
rows = []
for s in SEEDS:
    e, txt = cont(s, 28)
    rows.append((e, s, txt, bool(KNOWN.search(txt))))
rows.sort(key=lambda r: r[0])
log("=== lowest-entropy (most memorized) continuations ===")
log("    (*) = NOT a known decoy/loop attractor -> candidate hidden trigger\n")
for e, s, txt, known in rows:
    log(f"  H={e:4.2f} {'           ' if known else ' (*) NOVEL '} {s[:22]!r:24} -> {txt[:64]!r}")
log("\nDONE (attractor).")
